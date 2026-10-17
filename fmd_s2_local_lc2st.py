import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

import fmd_06_eval_real as ev
from fmd_02_make_datasets import DATA_ROOT, OUT_ROOT, PARAM_NAMES, load_config
from fmd_03_tune_npe import NPE_ROOT

SI_ROOT = OUT_ROOT / "si_local"
N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))
ALPHA = 0.05


def lc2st_all(model, seed, cfg, ref, theta_ref, x_local, z_eval, n_null, n_procs):
    de = ev.load_estimator(model, seed)
    s_ref = ev.embed(de, ev.encode(model, seed, ref["exp"], ref["rem"], cfg))
    s_obs = ev.embed(de, ev.encode(model, seed, cfg["exp_obs"], cfg["rem_obs"], cfg))
    s_loc = ev.embed(de, ev.encode(model, seed, x_local["exp"], x_local["rem"], cfg))
    mu, sd = s_ref.mean(axis=0), s_ref.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    s_ref_std = (s_ref - mu) / sd
    s_eval = (np.vstack([s_obs, s_loc]) - mu) / sd

    z_ref = ev.base_space(de, theta_ref, s_ref)
    jobs = [(k, ev.SEED + 1000 * seed + k) for k in range(n_null + 1)]
    with Pool(processes=n_procs, initializer=ev._init_lc2st, initargs=(z_ref, s_ref_std)) as pool:
        clfs = pool.map(ev.fit_classifier, jobs)

    t = np.array([[ev.lc2st_stat(c, z_eval, s) for s in s_eval] for c in clfs])
    t_obs, t_null = t[0], t[1:]
    p = (1 + (t_null >= t_obs[None, :]).sum(axis=0)) / (1 + n_null)
    return t_obs, t_null, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n_null", type=int, default=ev.N_NULL)
    parser.add_argument("--n_procs", type=int, default=N_CPUS)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.n_procs))
    cfg = load_config()
    ref = np.load(DATA_ROOT / "reference" / "reference.npz", allow_pickle=True)
    theta_ref = ref["theta"].astype(np.float64)
    loc = np.load(SI_ROOT / "local_datasets.npz", allow_pickle=True)
    x_local = {"exp": loc["exp"], "rem": loc["rem"]}
    n_sims = len(loc["exp"])
    z_eval = np.random.default_rng(ev.SEED).standard_normal((ev.N_LC2ST_EVAL, len(PARAM_NAMES)))

    shift = pd.read_csv(SI_ROOT / "local_check_long.csv")
    shift = (shift[shift["model"] != "mcmc"]
             .assign(abs_shift=lambda d: d["log_shift_vs_mcmc"].abs())
             .groupby(["model", "seed", "dataset"])["abs_shift"].mean())
    real_check = pd.read_csv(ev.EVAL_ROOT / "lc2st.csv") if (ev.EVAL_ROOT / "lc2st.csv").exists() else None

    rows, summ, nulls = [], [], {}
    for model in ev.MODELS:
        for s in args.seeds:
            if not (NPE_ROOT / f"seed_{s}" / f"{model}_density_estimator.pt").exists():
                continue
            t0 = time.perf_counter()
            t_obs, t_null, p = lc2st_all(model, s, cfg, ref, theta_ref, x_local, z_eval, args.n_null, args.n_procs)
            nulls[f"{model}_seed{s}"] = t_null
            labels = ["real"] + list(range(n_sims))
            for k, lab in enumerate(labels):
                rows.append({"model": model, "seed": s, "dataset": lab, "t_obs": t_obs[k], "p_value": p[k],
                             "null_q50": float(np.median(t_null[:, k])), "null_q95": float(np.quantile(t_null[:, k], 0.95))})
            t_loc, p_loc = t_obs[1:], p[1:]
            a = shift.loc[(model, s)].reindex(range(n_sims)).to_numpy()
            rho = float(pd.Series(t_loc).corr(pd.Series(a), method="spearman"))
            prev = np.nan
            if real_check is not None:
                hit = real_check[(real_check["model"] == model) & (real_check["seed"] == s)]
                prev = float(hit["t_obs"].iloc[0]) if len(hit) else np.nan
            summ.append({"model": model, "seed": s, "n_local": n_sims,
                         "local_reject_rate": float(np.mean(p_loc <= ALPHA)),
                         "local_p_median": float(np.median(p_loc)),
                         "local_t_median": float(np.median(t_loc)),
                         "real_t": float(t_obs[0]), "real_p": float(p[0]),
                         "real_t_percentile_in_local": float(np.mean(t_loc <= t_obs[0])),
                         "spearman_t_vs_abs_shift": rho,
                         "real_t_in_fmd06": prev})
            print(f"{model} seed {s}: local reject rate={summ[-1]['local_reject_rate']:.2f}  "
                  f"real p={p[0]:.3f}  real T pct={summ[-1]['real_t_percentile_in_local']:.2f}  "
                  f"rho(T,|shift|)={rho:.2f}  T real={t_obs[0]:.5f} (fmd_06 {prev:.5f})  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(SI_ROOT / "local_lc2st.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(SI_ROOT / "local_lc2st_summary.csv", index=False)
    np.savez_compressed(SI_ROOT / "local_lc2st_null.npz", **nulls)
    with open(SI_ROOT / "local_lc2st_meta.json", "w") as f:
        json.dump({"n_local": n_sims, "seeds": args.seeds, "n_null": args.n_null, "alpha": ALPHA,
                   "n_eval": ev.N_LC2ST_EVAL, "classifiers": "same reference-trained classifiers as fmd_06",
                   "abs_shift": "mean over parameters of |log NPE median - log MCMC median| from fmd_s1"}, f, indent=2)

    pd.set_option("display.width", 200)
    print(summ.drop(columns=["n_local"]).round(3).to_string(index=False), flush=True)
    print(f"Saved under {SI_ROOT}", flush=True)


if __name__ == "__main__":
    main()
