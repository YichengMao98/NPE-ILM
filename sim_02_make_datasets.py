import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
CONFIG_ROOT = Path(os.environ.get("ILM_CONFIG_ROOT", ARC_ROOT / "configs")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()

N_CPUS = max(1, int(os.environ.get("ILM_N_CPUS", os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
os.environ["OMP_NUM_THREADS"] = str(N_CPUS)
os.environ["MKL_NUM_THREADS"] = str(N_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPUS)

T = 40
IP_FIXED = 3
BASE_SEED = 1234

BETA_SHAPE = 6
BETA_SCALE = 1.0 / 4.0

R0_SHAPE = 10
R0_SCALE = 1.0 / 4.0

MU_D = math.log(3) - 0.27 ** 2 / 2
SIGMA_D = 0.27

RHO_EPS = 1e-6

PILOT_N_TRAIN = 3_000
PILOT_N_TEST = 300
FINAL_N_TRAIN = 20_000
FINAL_N_TEST = 1_000

PARAM_NAMES = {
    "full": ("alpha", "beta"),
    "partial": ("alpha", "beta", "rho"),
    "gamma": ("alpha", "beta", "gamma"),
}


def load_config():
    path = CONFIG_ROOT / "ilm_config.npz"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}\nRun sim_01_make_configs.py first.")
    dat = np.load(path, allow_pickle=True)
    return {
        "N": int(dat["N"]),
        "region_side": float(dat["region_side"]),
        "log_dist": dat["log_dist"].astype(np.float32),
        "seed_idx": dat["seed_idx"].astype(np.int32),
        "lut_betas": dat["lut_betas"].astype(np.float64),
        "lut_seed_rowsums": dat["lut_seed_rowsums"].astype(np.float64),
    }


def compute_sr_exact_np(betas, log_dist, seed_idx):
    betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    ld_seed = log_dist[seed_idx].astype(np.float64).copy()
    for k, sk in enumerate(seed_idx):
        ld_seed[k, sk] = np.inf
    n = len(betas)
    out = np.empty(n, dtype=np.float64)
    chunk = 500
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        bs = betas[s:e]
        W = np.exp(-bs[:, None, None] * ld_seed[None, :, :])
        out[s:e] = W.sum(axis=2).mean(axis=1)
    return out


def seed_rowsum_safe_np(beta, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    beta_arr = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    lut_lo, lut_hi = float(lut_betas[0]), float(lut_betas[-1])

    in_range = (beta_arr >= lut_lo) & (beta_arr <= lut_hi)
    out_range = ~in_range

    sr = np.empty_like(beta_arr)
    sr[in_range] = np.interp(beta_arr[in_range], lut_betas, lut_seed_rowsums)
    if out_range.any():
        sr[out_range] = compute_sr_exact_np(beta_arr[out_range], log_dist, seed_idx)

    return float(sr[0]) if beta_arr.shape[0] == 1 else sr


def draw_prior_full(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    beta = float(rng.gamma(BETA_SHAPE, BETA_SCALE))
    r0 = float(rng.gamma(R0_SHAPE, R0_SCALE))
    sr = seed_rowsum_safe_np(beta, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    alpha = float(r0 / (IP_FIXED * sr))
    return {"alpha": alpha, "beta": beta, "r0": r0}


def draw_prior_partial(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    pars = draw_prior_full(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    pars["rho"] = float(rng.uniform(RHO_EPS, 1.0 - RHO_EPS))
    return pars


def draw_prior_gamma(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    beta = float(rng.gamma(BETA_SHAPE, BETA_SCALE))
    r0 = float(rng.gamma(R0_SHAPE, R0_SCALE))
    log_d = rng.normal(MU_D, SIGMA_D)
    d = max(math.exp(log_d), 1.001)
    gamma = -math.log(1.0 - 1.0 / d)
    sr = seed_rowsum_safe_np(beta, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    alpha = float(r0 / (d * sr))
    return {"alpha": alpha, "beta": float(beta), "r0": float(r0), "d": float(d), "gamma": float(gamma)}


def simulate_fixed_ip(log_dist, alpha, beta, seed_idx, rng):
    N_loc = log_dist.shape[0]
    W = np.exp(-beta * log_dist.astype(np.float64))
    np.fill_diagonal(W, 0.0)
    infection_time = np.full(N_loc, -1, dtype=np.int32)
    infection_time[seed_idx] = 0
    for t in range(T):
        inf_mask = (infection_time >= 0) & ((t - infection_time) < IP_FIXED)
        susc_mask = infection_time < 0
        if not inf_mask.any():
            break
        pressure = alpha * (W @ inf_mask.astype(np.float64))
        p = np.clip(1.0 - np.exp(-pressure), 0.0, 1.0)
        p[~susc_mask] = 0.0
        new_inf = (rng.random(N_loc) < p) & susc_mask
        infection_time[new_inf] = t + 1
    return infection_time


def simulate_stochastic_recovery(log_dist, alpha, beta, gamma, seed_idx, rng):
    N_loc = log_dist.shape[0]
    p_ir = 1.0 - math.exp(-gamma)
    W = np.exp(-beta * log_dist.astype(np.float64))
    np.fill_diagonal(W, 0.0)
    infection_time = np.full(N_loc, -1, dtype=np.int32)
    recovered = np.zeros(N_loc, dtype=bool)
    infection_time[seed_idx] = 0

    for t in range(T):
        inf_mask = (infection_time >= 0) & (~recovered)
        susc_mask = infection_time < 0

        if not inf_mask.any():
            break

        pressure = alpha * (W @ inf_mask.astype(np.float64))
        p = np.clip(1.0 - np.exp(-pressure), 0.0, 1.0)
        p[~susc_mask] = 0.0
        new_inf = (rng.random(N_loc) < p) & susc_mask
        infection_time[new_inf] = t + 1

        inf_mask_after_transmission = (infection_time >= 0) & (~recovered)
        recovered[inf_mask_after_transmission & (rng.random(N_loc) < p_ir)] = True

    return infection_time


def epidemic_stats(infection_time, N_loc):
    n_inf = int((infection_time >= 0).sum())
    ar = n_inf / N_loc
    active = int(infection_time[infection_time >= 0].max()) if n_inf > 0 else 0
    return ar, active


def generate_dataset(scenario, n_keep, cfg, base_seed):
    N_loc = cfg["N"]
    log_dist = cfg["log_dist"]
    seed_idx = cfg["seed_idx"]
    lut_betas = cfg["lut_betas"]
    lut_seed_rowsums = cfg["lut_seed_rowsums"]

    rng = np.random.default_rng(base_seed)
    thetas, xs = [], []
    stats_rows = []
    n_oor = 0

    for i in range(n_keep):
        if scenario == "full":
            pars = draw_prior_full(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
            inf_t = simulate_fixed_ip(log_dist, pars["alpha"], pars["beta"], seed_idx, rng)
            theta = [pars["alpha"], pars["beta"]]
            x_store = inf_t.astype(np.int16)
            stats_row = {"alpha": pars["alpha"], "beta": pars["beta"], "r0": pars["r0"]}

        elif scenario == "partial":
            pars = draw_prior_partial(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
            inf_t = simulate_fixed_ip(log_dist, pars["alpha"], pars["beta"], seed_idx, rng)
            infected_idx = np.where(inf_t >= 0)[0]
            obs_mask = np.zeros(N_loc, dtype=bool)
            kept = rng.random(len(infected_idx)) < pars["rho"]
            obs_mask[infected_idx[kept]] = True
            obs_mask[seed_idx] = True
            inf_t_obs = inf_t.copy()
            inf_t_obs[~obs_mask & (inf_t >= 0)] = -1
            theta = [pars["alpha"], pars["beta"], pars["rho"]]
            x_store = inf_t_obs.astype(np.int16)
            stats_row = {"alpha": pars["alpha"], "beta": pars["beta"], "r0": pars["r0"], "rho": pars["rho"]}

        else:
            pars = draw_prior_gamma(rng, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
            inf_t = simulate_stochastic_recovery(log_dist, pars["alpha"], pars["beta"], pars["gamma"], seed_idx, rng)
            theta = [pars["alpha"], pars["beta"], pars["gamma"]]
            x_store = inf_t.astype(np.int16)
            stats_row = {
                "alpha": pars["alpha"],
                "beta": pars["beta"],
                "r0": pars["r0"],
                "d": pars["d"],
                "gamma": pars["gamma"],
            }

        if pars["beta"] < float(lut_betas[0]) or pars["beta"] > float(lut_betas[-1]):
            n_oor += 1

        ar, active = epidemic_stats(inf_t, N_loc)
        stats_row.update({"attack_rate": float(ar), "active_days": int(active)})
        thetas.append(theta)
        xs.append(x_store)
        stats_rows.append(stats_row)

        if (i + 1) % 1000 == 0:
            print(f"  generated {i + 1}/{n_keep}", flush=True)

    print(f"  done {n_keep} simulations (no rejection)")
    if n_oor > 0:
        print(f"  out-of-LUT-range betas (exact fallback used): "
              f"{n_oor}/{n_keep} ({n_oor / n_keep:.2%})")

    theta_arr = np.array(thetas, dtype=np.float32)
    x_arr = np.stack(xs, axis=0)
    stats_df = pd.DataFrame(stats_rows)
    return theta_arr, x_arr, stats_df


def save_split(out_dir, scenario, split, theta, x, stats_df, elapsed):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{split}.npz",
        theta=theta,
        x=x,
        T=np.int32(T),
        param_names=np.array(PARAM_NAMES[scenario], dtype=object),
    )
    stats_df.to_csv(out_dir / f"{split}_stats.csv", index=False)

    prior = {
        "beta": {"family": "Gamma", "shape": BETA_SHAPE, "scale": BETA_SCALE},
        "R0": {"family": "Gamma", "shape": R0_SHAPE, "scale": R0_SCALE},
        "alpha": {"family": "induced", "from": "R0 / (infectious_period * seed_rowsum(beta))"},
    }

    if scenario == "partial":
        prior["rho"] = {"family": "Uniform", "low": 0.0, "high": 1.0}

    if scenario == "gamma":
        prior["d"] = {"family": "LogNormal", "mu": MU_D, "sigma": SIGMA_D, "lower": 1.001}
        prior["gamma"] = {"family": "induced", "from": "-log(1 - 1 / d)"}

    with open(out_dir / f"{split}_meta.json", "w") as f:
        json.dump({
            "scenario": scenario,
            "split": split,
            "n": int(len(theta)),
            "n_params": int(theta.shape[1]),
            "param_names": list(PARAM_NAMES[scenario]),
            "T": int(T),
            "time_sec": float(elapsed),
            "prior": prior,
            "rejection_filter": "none",
        }, f, indent=2)

    print(f"  -> {split}.npz  theta={theta.shape}  x={x.shape}  ({elapsed:.1f}s)")


def run_phase(scenario, phase, n_train, n_test, cfg, out_root, seed_offset):
    print(f"\n[{scenario} / {phase}]  n_train={n_train}  n_test={n_test}")
    out_dir = out_root / scenario / phase

    t0 = time.perf_counter()
    th_tr, x_tr, st_tr = generate_dataset(scenario, n_train, cfg, BASE_SEED + seed_offset)
    save_split(out_dir, scenario, "train", th_tr, x_tr, st_tr, time.perf_counter() - t0)

    t0 = time.perf_counter()
    th_te, x_te, st_te = generate_dataset(scenario, n_test, cfg, BASE_SEED + seed_offset + 1)
    save_split(out_dir, scenario, "test", th_te, x_te, st_te, time.perf_counter() - t0)

    print(f"  alpha=[{st_tr['alpha'].min():.5f}, {st_tr['alpha'].max():.5f}]  "
          f"beta=[{st_tr['beta'].min():.3f}, {st_tr['beta'].max():.3f}]  "
          f"AR=[{st_tr['attack_rate'].min():.1%}, {st_tr['attack_rate'].max():.1%}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "partial", "gamma"])
    args = parser.parse_args()

    scenario = args.scenario
    cfg = load_config()

    seed_offsets = {"full": 0, "partial": 100, "gamma": 200}
    offset = seed_offsets[scenario]

    run_phase(scenario, "pilot", PILOT_N_TRAIN, PILOT_N_TEST, cfg, STRICT_ROOT, offset)
    run_phase(scenario, "final", FINAL_N_TRAIN, FINAL_N_TEST, cfg, STRICT_ROOT, offset + 10)

    print(f"\nDone. Outputs in {STRICT_ROOT / scenario}")


if __name__ == "__main__":
    main()