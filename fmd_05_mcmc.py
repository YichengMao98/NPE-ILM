import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import gamma as gamma_dist

from fmd_02_make_datasets import (
    BETA_SCALE,
    BETA_SHAPE,
    DATA_ROOT,
    OUT_ROOT,
    PARAM_NAMES,
    R0_SCALE,
    R0_SHAPE,
    draw_prior,
    lambda0,
    load_config,
    mean_ip,
)
from sim_05_mcmc_da import AdaptiveRW, Counter, accept
from sim_09_mcmc_diagnostics import ess_bulk, ess_tail, rhat_rank

MCMC_ROOT = OUT_ROOT / "mcmc"
N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))

SEED = 1234
N_CHAINS = 4
N_ITER = 100_000
N_BURNIN = 20_000
STORE_THIN = 10
COMPARE_DRAWS_PER_CHAIN = 750


def log1mexp(x):
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.log(-np.expm1(-np.maximum(x, 1e-300))), -np.inf)


def infectious_matrix(exp_obs, rem_obs, cfg):
    T = int(cfg["T"])
    L = int(cfg["latent"])
    n = int(cfg["N"])
    onset = np.full(n, 10**6, dtype=np.int64)
    rem = np.full(n, 10**6, dtype=np.int64)
    exp_obs = np.asarray(exp_obs, dtype=np.int64)
    rem_obs = np.asarray(rem_obs, dtype=np.int64)
    for key in ("active", "removed"):
        idx = cfg[f"{key}_idx"]
        onset[idx] = cfg[f"{key}_exp"] + L
        rem[idx] = cfg[f"{key}_rem"]
    idx = cfg["exposed_idx"]
    onset[idx] = cfg["exposed_exp"] + L
    rem[idx] = rem_obs[idx]
    w = np.where(exp_obs >= 1)[0]
    onset[w] = exp_obs[w] + L
    has = w[rem_obs[w] >= 0]
    rem[has] = rem_obs[has]
    need = np.r_[idx, w]
    if np.any((onset[need] <= T - 1) & (rem_obs[need] < 0)):
        raise ValueError("Removal time missing for a farm with infectious onset inside the window.")
    t = np.arange(T)[:, None]
    return ((onset[None, :] <= t) & (t < rem[None, :])).astype(np.float64)


class ExactLikelihood:
    def __init__(self, cfg, exp_obs, rem_obs):
        T = int(cfg["T"])
        self.sus = np.where(cfg["susceptible_mask"])[0]
        self.log_kernel = cfg["log_kernel"][:, self.sus].astype(np.float64)
        self.z = cfg["z"][self.sus].astype(np.float64)
        self.I = infectious_matrix(exp_obs, rem_obs, cfg)
        e = np.asarray(exp_obs, dtype=np.int64)[self.sus]
        t = np.arange(T)[:, None]
        self.hit = e[None, :] == t + 1
        self.miss = (e[None, :] < 0) | (e[None, :] > t + 1)
        self.beta = None
        self.S = None

    def base(self, beta):
        if beta != self.beta:
            self.S = self.I @ np.exp(-beta * self.log_kernel)
            self.beta = beta
        return self.S

    def __call__(self, alpha0, alpha1, beta):
        lam = (alpha0 + alpha1 * self.z)[None, :] * self.base(beta)
        return float(log1mexp(lam[self.hit]).sum() - lam[self.miss].sum())


def log_prior(theta, cfg):
    alpha0, alpha1, beta = (float(v) for v in theta)
    if alpha0 <= 0 or alpha1 <= 0 or beta <= 0:
        return -math.inf
    a_high = alpha0 + alpha1
    r = alpha0 / a_high
    scale = mean_ip(cfg) * lambda0(beta, r, cfg)
    r0 = a_high * scale
    return float(
        gamma_dist.logpdf(beta, a=BETA_SHAPE, scale=BETA_SCALE)
        + gamma_dist.logpdf(r0, a=R0_SHAPE, scale=R0_SCALE)
        + math.log(scale)
        - math.log(a_high)
    )


def load_target(target, cfg):
    if target == "real":
        return cfg["exp_obs"], cfg["rem_obs"], None
    idx = int(target.split(":")[1])
    ref = np.load(DATA_ROOT / "reference" / "reference.npz", allow_pickle=True)
    return ref["exp"][idx], ref["rem"][idx], ref["theta"][idx].astype(np.float64)


def run_chain(args):
    chain_id, target, n_iter, n_burnin, seed = args
    cfg = load_config()
    exp_obs, rem_obs, _ = load_target(target, cfg)
    lik = ExactLikelihood(cfg, exp_obs, rem_obs)
    rng = np.random.default_rng(seed)

    while True:
        theta, _, _ = draw_prior(rng, cfg)
        lp = log_prior(theta, cfg)
        ll = lik(*theta)
        if np.isfinite(lp) and np.isfinite(ll):
            break

    rw = AdaptiveRW(len(PARAM_NAMES), n_burnin)
    counter = Counter(("theta",))
    n_keep = n_iter - n_burnin
    samples = np.empty((n_keep, len(PARAM_NAMES)))
    loglik = np.empty(n_keep)
    logpost = np.empty(n_keep)
    x = np.log(theta)
    t0 = time.process_time()
    w0 = time.perf_counter()

    for it in range(n_iter):
        record = it >= n_burnin
        prop = rw.propose(x, rng)
        th_p = np.exp(prop)
        lp_p = log_prior(th_p, cfg)
        ok = False
        if np.isfinite(lp_p):
            ll_p = lik(*th_p)
            if np.isfinite(ll_p) and accept(rng, (ll_p + lp_p + prop.sum()) - (ll + lp + x.sum())):
                x, theta, ll, lp = prop, th_p, ll_p, lp_p
                ok = True
        counter.add("theta", ok, record)
        rw.update(x, it)
        if record:
            k = it - n_burnin
            samples[k] = theta
            loglik[k] = ll
            logpost[k] = ll + lp
        if (it + 1) % 10000 == 0:
            print(f"    chain {chain_id} iter {it + 1}/{n_iter}  ll={ll:.2f}  "
                  f"theta={np.array2string(theta, precision=5)}  {time.perf_counter() - w0:.0f}s", flush=True)

    return {
        "samples": samples,
        "loglik": loglik,
        "logpost": logpost,
        "acceptance": counter.rates(),
        "cpu_sec": float(time.process_time() - t0),
    }


def diagnostics(x):
    return {"rhat_rank": rhat_rank(x), "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="real")
    parser.add_argument("--n_chains", type=int, default=N_CHAINS)
    parser.add_argument("--n_iter", type=int, default=N_ITER)
    parser.add_argument("--n_burnin", type=int, default=N_BURNIN)
    parser.add_argument("--store_thin", type=int, default=STORE_THIN)
    parser.add_argument("--draws_per_chain", type=int, default=COMPARE_DRAWS_PER_CHAIN)
    parser.add_argument("--n_procs", type=int, default=N_CPUS)
    args = parser.parse_args()

    if args.target != "real" and not args.target.startswith("ref:"):
        raise ValueError("--target must be 'real' or 'ref:<index>'")
    if args.n_burnin >= args.n_iter:
        raise ValueError("n_burnin must be smaller than n_iter")

    cfg = load_config()
    _, _, true_theta = load_target(args.target, cfg)
    tag = "real" if args.target == "real" else f"ref_{int(args.target.split(':')[1]):05d}"
    MCMC_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"target={args.target}  chains={args.n_chains}  iter={args.n_iter}  burnin={args.n_burnin}  "
          f"store_thin={args.store_thin}  procs={args.n_procs}", flush=True)
    if true_theta is not None:
        print("  true: " + "  ".join(f"{p}={v:.5g}" for p, v in zip(PARAM_NAMES, true_theta)), flush=True)

    rng_init = np.random.default_rng(SEED + 7)
    chain_args = [(c, args.target, args.n_iter, args.n_burnin, int(rng_init.integers(1, 10**9)))
                  for c in range(args.n_chains)]
    w0 = time.perf_counter()
    with Pool(processes=max(1, min(args.n_chains, args.n_procs))) as pool:
        results = pool.map(run_chain, chain_args)
    wall = time.perf_counter() - w0

    samples = np.stack([r["samples"] for r in results])
    loglik = np.stack([r["loglik"] for r in results])
    logpost = np.stack([r["logpost"] for r in results])
    cpu = float(sum(r["cpu_sec"] for r in results))

    diag = {p: diagnostics(samples[:, :, j]) for j, p in enumerate(PARAM_NAMES)}
    diag["loglik"] = diagnostics(loglik)
    diag["logpost"] = diagnostics(logpost)

    n_keep = samples.shape[1]
    pick = np.unique(np.linspace(0, n_keep - 1, min(args.draws_per_chain, n_keep)).round().astype(int))
    keep = slice(None, None, args.store_thin)
    arrays = dict(
        samples=samples[:, keep, :].astype(np.float32),
        loglik=loglik[:, keep].astype(np.float32),
        logpost=logpost[:, keep].astype(np.float32),
        samples_compare=samples[:, pick, :].reshape(-1, len(PARAM_NAMES)).astype(np.float32),
        param_names=np.array(PARAM_NAMES, dtype=object),
        n_chains=np.int32(args.n_chains),
        n_keep=np.int32(n_keep),
        store_thin=np.int32(args.store_thin),
    )
    if true_theta is not None:
        arrays["theta"] = true_theta.astype(np.float32)
    np.savez_compressed(MCMC_ROOT / f"{tag}.npz", **arrays)

    meta = {
        "target": args.target,
        "sampler": "fmd_05_mcmc exact likelihood",
        "param_names": list(PARAM_NAMES),
        "proposal": "adaptive random walk on (log alpha0, log alpha1, log beta), adapted during burn-in only",
        "n_chains": int(args.n_chains),
        "n_iter": int(args.n_iter),
        "n_burnin": int(args.n_burnin),
        "n_keep": int(n_keep),
        "store_thin": int(args.store_thin),
        "n_compare_per_chain": int(len(pick)),
        "diagnostics_on": "all post-burn-in draws",
        "timing": {"wall_sec": float(wall), "cpu_sec": cpu},
        "acceptance": [r["acceptance"] for r in results],
        "diagnostics": diag,
    }
    with open(MCMC_ROOT / f"{tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    flat = samples.reshape(-1, len(PARAM_NAMES))
    med = np.median(flat, axis=0)
    lo, hi = np.quantile(flat, [0.025, 0.975], axis=0)
    print(f"  wall={wall:.0f}s  cpu={cpu:.0f}s", flush=True)
    for j, p in enumerate(PARAM_NAMES):
        print(f"  {p:>8}: median={med[j]:.5g}  95% CI=[{lo[j]:.5g}, {hi[j]:.5g}]", flush=True)
    for name, d in diag.items():
        print(f"  {name:>8}: R-hat={d['rhat_rank']:.4f}  ESS bulk={d['ess_bulk']:.0f}  tail={d['ess_tail']:.0f}", flush=True)
    for c, r in enumerate(results):
        print(f"  chain {c} acceptance={r['acceptance']['theta']:.3f}", flush=True)
    print(f"Saved: {MCMC_ROOT / (tag + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
