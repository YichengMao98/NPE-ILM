import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sbi.inference.posteriors.direct_posterior import DirectPosterior

from fmd_02_make_datasets import OUT_ROOT, PARAM_NAMES, draw_prior, load_config, simulate
from fmd_03_tune_npe import FMDPrior, sample_valid
from fmd_05_mcmc import ExactLikelihood, log_prior
from fmd_06_eval_real import encode, load_estimator
from sim_05_mcmc_da import AdaptiveRW, accept
from sim_09_mcmc_diagnostics import ess_bulk, rhat_rank

MCMC_ROOT = OUT_ROOT / "mcmc"
SI_ROOT = OUT_ROOT / "si_local"
N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))

SEED = 2468
MODELS = ("cnn", "gnn")
N_SIMS = 50
N_CHAINS = 2
N_ITER = 20_000
N_BURNIN = 5_000
N_POST = 2000

_CFG = None


def _init():
    global _CFG
    _CFG = load_config()
    _CFG["log_kernel"] = _CFG["log_kernel"].astype(np.float64)


def make_datasets(n_sims, rng):
    post = np.load(MCMC_ROOT / "real.npz", allow_pickle=True)["samples_compare"].astype(np.float64)
    pick = rng.choice(len(post), size=n_sims, replace=False)
    theta = post[pick]
    cfg = load_config()
    lk = cfg["log_kernel"].astype(np.float64)
    exp = np.empty((n_sims, int(cfg["N"])), dtype=np.int16)
    rem = np.empty_like(exp)
    for k in range(n_sims):
        exp[k], rem[k] = simulate(theta[k], cfg, rng, lk)
    return theta, exp, rem


def mcmc_chain(args):
    k, c, exp, rem, n_iter, n_burnin, seed = args
    rng = np.random.default_rng(seed)
    lik = ExactLikelihood(_CFG, exp, rem)
    while True:
        theta, _, _ = draw_prior(rng, _CFG)
        lp = log_prior(theta, _CFG)
        ll = lik(*theta)
        if np.isfinite(lp) and np.isfinite(ll):
            break
    rw = AdaptiveRW(len(PARAM_NAMES), n_burnin)
    x = np.log(theta)
    out = np.empty((n_iter - n_burnin, len(PARAM_NAMES)))
    for it in range(n_iter):
        prop = rw.propose(x, rng)
        th_p = np.exp(prop)
        lp_p = log_prior(th_p, _CFG)
        if np.isfinite(lp_p):
            ll_p = lik(*th_p)
            if np.isfinite(ll_p) and accept(rng, (ll_p + lp_p + prop.sum()) - (ll + lp + x.sum())):
                x, theta, ll, lp = prop, th_p, ll_p, lp_p
        rw.update(x, it)
        if it >= n_burnin:
            out[it - n_burnin] = theta
    return k, c, out


def run_mcmc(exp, rem, n_chains, n_iter, n_burnin, n_procs, rng):
    jobs = [(k, c, exp[k], rem[k], n_iter, n_burnin, int(rng.integers(1, 10**9)))
            for k in range(len(exp)) for c in range(n_chains)]
    with Pool(processes=n_procs, initializer=_init) as pool:
        res = pool.map(mcmc_chain, jobs)
    chains = {}
    for k, c, out in res:
        chains.setdefault(k, {})[c] = out
    return [np.stack([chains[k][c] for c in range(n_chains)]) for k in range(len(exp))]


def npe_samples(model, seed, exp, rem, cfg, prior, n_post):
    de = load_estimator(model, seed)
    posterior = DirectPosterior(posterior_estimator=de, prior=prior)
    x = encode(model, seed, exp, rem, cfg)
    out = []
    for k in range(len(x)):
        draw = sample_valid(posterior, torch.tensor(x[k:k + 1], dtype=torch.float32), n_post)
        out.append(np.full((n_post, len(PARAM_NAMES)), np.nan) if draw is None else draw.astype(np.float64))
    return out


def summarize(samples, truth):
    lo, med, hi = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
    return med, lo, hi, (truth >= lo) & (truth <= hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims", type=int, default=N_SIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n_chains", type=int, default=N_CHAINS)
    parser.add_argument("--n_iter", type=int, default=N_ITER)
    parser.add_argument("--n_burnin", type=int, default=N_BURNIN)
    parser.add_argument("--n_post", type=int, default=N_POST)
    parser.add_argument("--n_procs", type=int, default=N_CPUS)
    args = parser.parse_args()

    SI_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.n_procs))
    rng = np.random.default_rng(SEED)
    cfg = load_config()
    prior = FMDPrior(cfg)

    t0 = time.perf_counter()
    theta, exp, rem = make_datasets(args.n_sims, rng)
    np.savez_compressed(SI_ROOT / "local_datasets.npz", theta=theta.astype(np.float32), exp=exp, rem=rem,
                        param_names=np.array(PARAM_NAMES, dtype=object))
    print(f"simulated {args.n_sims} datasets from the MCMC posterior  "
          f"exposures median={np.median((exp >= 1).sum(axis=1)):.0f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    mcmc = run_mcmc(exp, rem, args.n_chains, args.n_iter, args.n_burnin, args.n_procs, rng)
    print(f"MCMC done ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows, diag = [], []
    for k in range(args.n_sims):
        ch = mcmc[k]
        diag.append({"dataset": k, **{f"rhat_{p}": rhat_rank(ch[:, :, j]) for j, p in enumerate(PARAM_NAMES)},
                     **{f"ess_bulk_{p}": ess_bulk(ch[:, :, j]) for j, p in enumerate(PARAM_NAMES)}})
        med, lo, hi, cov = summarize(ch.reshape(-1, len(PARAM_NAMES)), theta[k])
        for j, p in enumerate(PARAM_NAMES):
            rows.append({"dataset": k, "model": "mcmc", "seed": 0, "parameter": p, "true": theta[k, j],
                         "median": med[j], "q025": lo[j], "q975": hi[j], "covered": bool(cov[j])})

    for model in MODELS:
        for s in args.seeds:
            if not (OUT_ROOT / "npe" / f"seed_{s}" / f"{model}_density_estimator.pt").exists():
                continue
            t0 = time.perf_counter()
            draws = npe_samples(model, s, exp, rem, cfg, prior, args.n_post)
            for k in range(args.n_sims):
                med, lo, hi, cov = summarize(draws[k], theta[k])
                for j, p in enumerate(PARAM_NAMES):
                    rows.append({"dataset": k, "model": model, "seed": s, "parameter": p, "true": theta[k, j],
                                 "median": med[j], "q025": lo[j], "q975": hi[j], "covered": bool(cov[j])})
            print(f"{model} seed {s} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["log_error"] = np.log(df["median"]) - np.log(df["true"])
    df["log_width"] = np.log(df["q975"]) - np.log(df["q025"])
    ref = df[df["model"] == "mcmc"][["dataset", "parameter", "median"]].rename(columns={"median": "mcmc_median"})
    df = df.merge(ref, on=["dataset", "parameter"])
    df["log_shift_vs_mcmc"] = np.log(df["median"]) - np.log(df["mcmc_median"])
    df.to_csv(SI_ROOT / "local_check_long.csv", index=False)
    pd.DataFrame(diag).to_csv(SI_ROOT / "local_check_mcmc_diagnostics.csv", index=False)

    summ = (df.groupby(["model", "parameter"])
              .agg(mean_log_error=("log_error", "mean"), sd_log_error=("log_error", "std"),
                   coverage=("covered", "mean"), mean_log_width=("log_width", "mean"),
                   mean_log_shift_vs_mcmc=("log_shift_vs_mcmc", "mean"),
                   sd_log_shift_vs_mcmc=("log_shift_vs_mcmc", "std"))
              .reset_index())

    real = {}
    rp = np.load(MCMC_ROOT / "real.npz", allow_pickle=True)["samples_compare"].astype(np.float64)
    real_mcmc = np.median(rp, axis=0)
    for model in MODELS:
        meds = []
        for s in args.seeds:
            p = OUT_ROOT / "npe" / f"seed_{s}" / f"{model}_real_posterior.npz"
            if p.exists():
                meds.append(np.log(np.median(np.load(p)["samples"].astype(np.float64), axis=0)) - np.log(real_mcmc))
        if meds:
            real[model] = np.mean(meds, axis=0)
    pct = []
    for _, r in summ.iterrows():
        if r["model"] in real:
            j = PARAM_NAMES.index(r["parameter"])
            local = df[(df["model"] == r["model"]) & (df["parameter"] == r["parameter"])]
            per_ds = local.groupby("dataset")["log_shift_vs_mcmc"].mean().to_numpy()
            pct.append((real[r["model"]][j], float(np.mean(per_ds <= real[r["model"]][j]))))
        else:
            pct.append((np.nan, np.nan))
    summ["real_log_shift_vs_mcmc"] = [v[0] for v in pct]
    summ["real_shift_percentile_in_local"] = [v[1] for v in pct]
    summ.to_csv(SI_ROOT / "local_check_summary.csv", index=False)

    with open(SI_ROOT / "local_check_meta.json", "w") as f:
        json.dump({"n_sims": args.n_sims, "theta_source": "draws from the real-data MCMC posterior",
                   "seeds": args.seeds, "mcmc": {"n_chains": args.n_chains, "n_iter": args.n_iter,
                                                 "n_burnin": args.n_burnin},
                   "n_post": args.n_post, "max_rhat": float(pd.DataFrame(diag)[[f"rhat_{p}" for p in PARAM_NAMES]].max().max())},
                  f, indent=2)

    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False), flush=True)
    print(f"max MCMC rank R-hat across datasets: {pd.DataFrame(diag)[[f'rhat_{p}' for p in PARAM_NAMES]].max().max():.4f}")
    print(f"Saved under {SI_ROOT}", flush=True)


if __name__ == "__main__":
    main()
