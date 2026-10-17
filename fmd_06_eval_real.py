import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from fmd_02_make_datasets import DATA_ROOT, OUT_ROOT, PARAM_NAMES, draw_prior, load_config, simulate
from fmd_03_tune_npe import NPE_ROOT, daily_counts, encode_cnn, encode_gnn

MCMC_ROOT = OUT_ROOT / "mcmc"
EVAL_ROOT = OUT_ROOT / "eval"
N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))

SEED = 4321
MODELS = ("cnn", "gnn")
PARTS = ("posterior", "lc2st", "mmd", "ppc", "mcmc")
N_PPC = 500
N_NULL = 50
N_LC2ST_EVAL = 10000
EMB_BATCH = 256

_CFG = None


def load_posteriors(seeds):
    post = {}
    for s in seeds:
        for m in MODELS:
            p = NPE_ROOT / f"seed_{s}" / f"{m}_real_posterior.npz"
            if p.exists():
                post[(m, s)] = np.load(p, allow_pickle=True)["samples"].astype(np.float64)
    mp = MCMC_ROOT / "real.npz"
    if mp.exists():
        post[("mcmc", 0)] = np.load(mp, allow_pickle=True)["samples_compare"].astype(np.float64)
    return post


def summarize(samples):
    q = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
    return q, samples.mean(axis=0), samples.std(axis=0, ddof=1)


def part_posterior(post, seeds):
    rows = []
    for (m, s), smp in post.items():
        q, mean, sd = summarize(smp)
        for j, p in enumerate(PARAM_NAMES):
            rows.append({"model": m, "seed": s, "parameter": p, "median": q[1, j], "q025": q[0, j],
                         "q975": q[2, j], "mean": mean[j], "sd": sd[j]})
    for m in MODELS:
        pooled = [post[(m, s)] for s in seeds if (m, s) in post]
        if not pooled:
            continue
        q, mean, sd = summarize(np.concatenate(pooled))
        for j, p in enumerate(PARAM_NAMES):
            rows.append({"model": m, "seed": "pooled", "parameter": p, "median": q[1, j], "q025": q[0, j],
                         "q975": q[2, j], "mean": mean[j], "sd": sd[j]})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_ROOT / "posterior_summary.csv", index=False)

    var_rows = []
    ref_sd = post[("mcmc", 0)].std(axis=0, ddof=1) if ("mcmc", 0) in post else None
    for m in MODELS:
        per = df[(df["model"] == m) & (df["seed"] != "pooled")]
        if per.empty:
            continue
        for j, p in enumerate(PARAM_NAMES):
            d = per[per["parameter"] == p]
            sd_med = float(d["median"].std(ddof=1)) if len(d) > 1 else float("nan")
            var_rows.append({
                "model": m, "parameter": p, "n_seeds": int(len(d)),
                "median_of_medians": float(d["median"].median()),
                "sd_of_medians": sd_med,
                "mean_ci_width": float((d["q975"] - d["q025"]).mean()),
                "sd_of_medians_over_mcmc_sd": sd_med / ref_sd[j] if ref_sd is not None else float("nan"),
            })
    pd.DataFrame(var_rows).to_csv(EVAL_ROOT / "seed_variability.csv", index=False)
    print(f"posterior summaries: {len(post)} posterior sets", flush=True)


def load_estimator(model, seed):
    de = torch.load(NPE_ROOT / f"seed_{seed}" / f"{model}_density_estimator.pt", map_location="cpu", weights_only=False)
    de.eval()
    return de


def encode(model, seed, exp, rem, cfg):
    if model == "gnn":
        return encode_gnn(exp, rem, cfg)
    path = NPE_ROOT / f"seed_{seed}" / "cnn_norm.npz"
    norm = np.load(path if path.exists() else NPE_ROOT / "cnn_norm.npz")
    return encode_cnn(exp, rem, cfg, norm["mean"], norm["std"])


@torch.no_grad()
def embed(de, x):
    out = []
    for i in range(0, len(x), EMB_BATCH):
        out.append(de._embedding_net(torch.tensor(x[i:i + EMB_BATCH], dtype=torch.float32)).numpy())
    return np.concatenate(out).astype(np.float64)


@torch.no_grad()
def base_space(de, theta, emb):
    z, _ = de._transform(torch.tensor(theta, dtype=torch.float32), context=torch.tensor(emb, dtype=torch.float32))
    return z.numpy().astype(np.float64)


def reference_embeddings(model, seed, cfg, ref):
    de = load_estimator(model, seed)
    x_ref = encode(model, seed, ref["exp"], ref["rem"], cfg)
    x_obs = encode(model, seed, cfg["exp_obs"], cfg["rem_obs"], cfg)
    s_ref = embed(de, x_ref)
    s_obs = embed(de, x_obs)
    return de, s_ref, s_obs


_Z_REF = None
_S_REF = None


def _init_lc2st(z_ref, s_ref):
    global _Z_REF, _S_REF
    _Z_REF, _S_REF = z_ref, s_ref


def fit_classifier(args):
    k, seed = args
    rng = np.random.default_rng(seed)
    z1 = _Z_REF if k == 0 else rng.standard_normal(_Z_REF.shape)
    z0 = rng.standard_normal(_Z_REF.shape)
    feats = np.vstack([np.hstack([z1, _S_REF]), np.hstack([z0, _S_REF])])
    labels = np.r_[np.ones(len(z1)), np.zeros(len(z0))]
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True, random_state=seed)
    clf.fit(feats, labels)
    return clf


def sq_dist(a, b):
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def mmd_point(q, bank, h2):
    return 1.0 - 2.0 * np.exp(-sq_dist(q, bank) / h2).mean(axis=1)


def lc2st_stat(clf, z_eval, s_obs_std):
    feats = np.hstack([z_eval, np.repeat(s_obs_std[None, :], len(z_eval), axis=0)])
    p = clf.predict_proba(feats)[:, 1]
    return float(np.mean((p - 0.5) ** 2))


def part_lc2st_mmd(seeds, parts, n_null, n_procs):
    cfg = load_config()
    ref = np.load(DATA_ROOT / "reference" / "reference.npz", allow_pickle=True)
    theta_ref = ref["theta"].astype(np.float64)
    rng = np.random.default_rng(SEED)
    z_eval = rng.standard_normal((N_LC2ST_EVAL, len(PARAM_NAMES)))
    lc_rows, mmd_rows, null_store = [], [], {}

    for s in seeds:
        for m in MODELS:
            path = NPE_ROOT / f"seed_{s}" / f"{m}_density_estimator.pt"
            if not path.exists():
                continue
            t0 = time.perf_counter()
            de, s_ref, s_obs = reference_embeddings(m, s, cfg, ref)
            mu, sd = s_ref.mean(axis=0), s_ref.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            s_ref_std = (s_ref - mu) / sd
            s_obs_std = ((s_obs - mu) / sd)[0]

            if "lc2st" in parts:
                z_ref = base_space(de, theta_ref, s_ref)
                jobs = [(k, SEED + 1000 * s + k) for k in range(n_null + 1)]
                with Pool(processes=n_procs, initializer=_init_lc2st, initargs=(z_ref, s_ref_std)) as pool:
                    clfs = pool.map(fit_classifier, jobs)
                t_obs = lc2st_stat(clfs[0], z_eval, s_obs_std)
                t_null = np.array([lc2st_stat(c, z_eval, s_obs_std) for c in clfs[1:]])
                p_val = float((1 + np.sum(t_null >= t_obs)) / (1 + len(t_null)))
                null_store[f"lc2st_null_{m}_seed{s}"] = t_null
                lc_rows.append({"model": m, "seed": s, "t_obs": t_obs, "p_value": p_val,
                                "null_q50": float(np.median(t_null)), "null_q95": float(np.quantile(t_null, 0.95)),
                                **{f"z_mean_{p}": float(v) for p, v in zip(PARAM_NAMES, z_ref.mean(axis=0))},
                                **{f"z_sd_{p}": float(v) for p, v in zip(PARAM_NAMES, z_ref.std(axis=0))}})
                print(f"  L-C2ST {m} seed {s}: T={t_obs:.5f}  p={p_val:.3f}", flush=True)

            if "mmd" in parts:
                half = len(s_ref_std) // 2
                bank, queries = s_ref_std[:half], s_ref_std[half:]
                sub = bank[rng.choice(len(bank), size=min(1000, len(bank)), replace=False)]
                h2 = float(np.median(sq_dist(sub, sub)[np.triu_indices(len(sub), 1)]))
                null = np.concatenate([mmd_point(queries[i:i + 200], bank, h2) for i in range(0, len(queries), 200)])
                obs = float(mmd_point(s_obs_std[None, :], bank, h2)[0])
                center = bank.mean(axis=0)
                prec = np.linalg.inv(np.cov(bank, rowvar=False) + 1e-6 * np.eye(bank.shape[1]))
                m_null = np.einsum("ij,jk,ik->i", queries - center, prec, queries - center)
                m_obs = float(np.einsum("j,jk,k->", s_obs_std - center, prec, s_obs_std - center))
                null_store[f"mmd_null_{m}_seed{s}"] = null
                mmd_rows.append({"model": m, "seed": s, "n_bank": int(len(bank)), "n_null": int(len(queries)),
                                 "mmd_obs": obs,
                                 "mmd_p_value": float((1 + np.sum(null >= obs)) / (1 + len(null))),
                                 "mahalanobis_obs": m_obs,
                                 "mahalanobis_p_value": float((1 + np.sum(m_null >= m_obs)) / (1 + len(m_null))),
                                 "bandwidth_sq": h2})
                print(f"  MMD {m} seed {s}: p={mmd_rows[-1]['mmd_p_value']:.4f}  "
                      f"Mahalanobis p={mmd_rows[-1]['mahalanobis_p_value']:.4f}", flush=True)
            print(f"    ({time.perf_counter() - t0:.0f}s)", flush=True)

    if lc_rows:
        pd.DataFrame(lc_rows).to_csv(EVAL_ROOT / "lc2st.csv", index=False)
    if mmd_rows:
        pd.DataFrame(mmd_rows).to_csv(EVAL_ROOT / "embedding_mmd.csv", index=False)
    if null_store:
        np.savez_compressed(EVAL_ROOT / "null_distributions.npz", **null_store)


def _init_ppc():
    global _CFG
    _CFG = load_config()
    _CFG["log_kernel"] = _CFG["log_kernel"].astype(np.float64)


def _ppc_chunk(args):
    thetas, seed = args
    rng = np.random.default_rng(seed)
    exp = np.empty((len(thetas), int(_CFG["N"])), dtype=np.int16)
    rem = np.empty_like(exp)
    for i, th in enumerate(thetas):
        exp[i], rem[i] = simulate(th, _CFG, rng, _CFG["log_kernel"])
    return exp, rem


def simulate_many(thetas, seed, n_procs):
    chunks = np.array_split(thetas, max(1, min(len(thetas), n_procs * 4)))
    seqs = np.random.SeedSequence(seed).spawn(len(chunks))
    with Pool(processes=n_procs, initializer=_init_ppc) as pool:
        parts = pool.map(_ppc_chunk, list(zip(chunks, seqs)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def auc(target, score):
    return float(roc_auc_score(target, score)) if 0 < target.sum() < len(target) else float("nan")


def part_ppc(post, n_ppc, n_procs):
    cfg = load_config()
    T = int(cfg["T"])
    obs = daily_counts(cfg["exp_obs"], cfg["rem_obs"], cfg)[0]
    obs_curves = np.stack([obs[0] + obs[1], obs[0], obs[1], obs[2]])
    sus = cfg["susceptible_mask"].astype(bool)
    z1 = cfg["z"].astype(bool)
    target = (cfg["exp_obs"] >= 1).astype(int)

    rng = np.random.default_rng(SEED)
    sets = dict(post)
    sets[("prior", 0)] = np.stack([draw_prior(rng, cfg)[0] for _ in range(n_ppc)])

    rows = []
    for k, (key, smp) in enumerate(sets.items()):
        m, s = key
        pick = rng.choice(len(smp), size=min(n_ppc, len(smp)), replace=False)
        exp, rem = simulate_many(smp[pick], SEED + 100 + k, n_procs)
        c = daily_counts(exp, rem, cfg)
        curves = np.stack([c[:, 0] + c[:, 1], c[:, 0], c[:, 1], c[:, 2]], axis=1)
        farm_prob = (exp >= 1).mean(axis=0)
        np.savez_compressed(EVAL_ROOT / f"ppc_{m}_seed{s}.npz",
                            exposures=curves[:, 0].astype(np.int16), exposures_z0=curves[:, 1].astype(np.int16),
                            exposures_z1=curves[:, 2].astype(np.int16), removals=curves[:, 3].astype(np.int16),
                            farm_prob=farm_prob.astype(np.float32), theta=smp[pick].astype(np.float32))
        lo, hi = np.quantile(curves, [0.025, 0.975], axis=0)
        inside = (obs_curves >= lo) & (obs_curves <= hi)
        row = {"model": m, "seed": s, "n_sims": len(pick),
               "exposure_days_in_95pi": float(inside[0].mean()),
               "exposure_z0_days_in_95pi": float(inside[1].mean()),
               "exposure_z1_days_in_95pi": float(inside[2].mean()),
               "removal_days_in_95pi": float(inside[3, :T - 1].mean())}
        for name, j in (("total", 0), ("z0", 1), ("z1", 2)):
            tot = curves[:, j].sum(axis=1)
            tot_obs = float(obs_curves[j].sum())
            row.update({f"{name}_exposures_obs": tot_obs,
                        f"{name}_exposures_median": float(np.median(tot)),
                        f"{name}_exposures_q025": float(np.quantile(tot, 0.025)),
                        f"{name}_exposures_q975": float(np.quantile(tot, 0.975)),
                        f"{name}_exposures_ppp": float(np.mean(tot >= tot_obs))})
        p = farm_prob[sus]
        y = target[sus]
        row.update({"farm_brier": float(np.mean((p - y) ** 2)),
                    "farm_auc": auc(y, p),
                    "farm_auc_z0": auc(target[sus & ~z1], farm_prob[sus & ~z1]),
                    "farm_auc_z1": auc(target[sus & z1], farm_prob[sus & z1])})
        rows.append(row)
        print(f"  PPC {m} seed {s}: exposure days in 95% PI={row['exposure_days_in_95pi']:.2f}  "
              f"total exposures median={row['total_exposures_median']:.0f} (obs {row['total_exposures_obs']:.0f}, "
              f"ppp={row['total_exposures_ppp']:.2f})  AUC={row['farm_auc']:.3f} "
              f"(z0 {row['farm_auc_z0']:.3f}, z1 {row['farm_auc_z1']:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(EVAL_ROOT / "ppc_summary.csv", index=False)
    np.savez_compressed(EVAL_ROOT / "ppc_observed.npz", exposures=obs_curves[0], exposures_z0=obs_curves[1],
                        exposures_z1=obs_curves[2], removals=obs_curves[3], exposed=target.astype(bool),
                        susceptible=sus, z=cfg["z"], positions=cfg["positions"], T=np.int32(T))


def part_mcmc():
    path = MCMC_ROOT / "real_meta.json"
    if not path.exists():
        print("mcmc skipped: real_meta.json not found", flush=True)
        return
    meta = json.load(open(path))
    rows = [{"quantity": q, **d} for q, d in meta["diagnostics"].items()]
    df = pd.DataFrame(rows)
    df["acceptance_mean"] = float(np.mean([a["theta"] for a in meta["acceptance"]]))
    df["wall_sec"] = meta["timing"]["wall_sec"]
    df.to_csv(EVAL_ROOT / "mcmc_diagnostics.csv", index=False)
    print(df[["quantity", "rhat_rank", "ess_bulk", "ess_tail"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    parser.add_argument("--n_ppc", type=int, default=N_PPC)
    parser.add_argument("--n_null", type=int, default=N_NULL)
    parser.add_argument("--n_procs", type=int, default=N_CPUS)
    args = parser.parse_args()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.n_procs))
    post = load_posteriors(args.seeds)
    print(f"loaded posteriors: {sorted(map(str, post.keys()))}", flush=True)

    if "mcmc" in args.parts:
        part_mcmc()
    if "posterior" in args.parts:
        part_posterior(post, args.seeds)
    if "lc2st" in args.parts or "mmd" in args.parts:
        part_lc2st_mmd(args.seeds, args.parts, args.n_null, args.n_procs)
    if "ppc" in args.parts:
        part_ppc(post, args.n_ppc, args.n_procs)
    print(f"Saved under {EVAL_ROOT}", flush=True)


if __name__ == "__main__":
    main()
