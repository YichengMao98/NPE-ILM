import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sim_06_eval_cnnvsgnn import (
    IP_FIXED,
    load_config,
    inf_time_to_daily,
    simulate_fixed_ip_curve,
    simulate_gamma_curve,
    apply_partial_observation,
)

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = ARC_ROOT / "strict"
SEEDS_ROOT = ARC_ROOT / "strict_seeds"
DIAG_ROOT = ARC_ROOT / "mcmc_diagnostics"
OUT_ROOT = ARC_ROOT / "individual_ppc"

DEFAULT_SEED = 1
MIN_OBS_INF = 50
QUANTILES = (0.5,)
N_SEL_DRAWS = 200
SIM_SEED = 20261010
METHODS = ("CNN-NPE", "GNN-NPE", "MCMC")
METHOD_KEYS = {"CNN-NPE": "cnn", "GNN-NPE": "gnn", "MCMC": "mcmc"}


def load_inputs(scenario, seed):
    final_dir = SEEDS_ROOT / f"seed_{seed}" / scenario / "final"
    te = np.load(final_dir / "test.npz", allow_pickle=True)
    x = te["x"].astype(np.int32)
    theta = te["theta"].astype(np.float64)
    param_names = tuple(te["param_names"].tolist())
    T = int(te["T"])
    npe = {m: np.load(final_dir / f"{METHOD_KEYS[m]}_posterior_samples.npz")["samples"].astype(np.float64)
           for m in ("CNN-NPE", "GNN-NPE")}
    mcmc_dir = STRICT_ROOT / scenario / "mcmc"
    raw = np.load(mcmc_dir / "mcmc_samples.npz", allow_pickle=True)["samples"].astype(np.float64)
    mcmc = raw.reshape(raw.shape[0], -1, raw.shape[-1])
    obs_idx = np.load(mcmc_dir / "mcmc_obs_idx.npy").astype(int)
    diag = pd.read_csv(DIAG_ROOT / f"diagnostics_by_obs_{scenario}.csv").sort_values("local_mcmc_index")
    if not np.array_equal(diag["obs_idx"].to_numpy(), obs_idx):
        raise ValueError(f"Diagnostics do not match MCMC observations for {scenario}")
    if not np.allclose(theta[obs_idx], np.load(mcmc_dir / "mcmc_theta_test.npy"), rtol=1e-5, atol=1e-6):
        raise ValueError(f"MCMC theta does not match test theta for {scenario}")
    return {
        "x": x, "theta": theta, "param_names": param_names, "T": T,
        "npe": npe, "mcmc": mcmc, "obs_idx": obs_idx,
        "converged": diag["converged"].to_numpy(bool),
    }


def method_samples(data, method, k):
    if method == "MCMC":
        return data["mcmc"][k]
    return data["npe"][method][data["obs_idx"][k]]


def simulate(scenario, pars, cfg, T, rng):
    if scenario == "full":
        return simulate_fixed_ip_curve(cfg["log_dist"], float(pars[0]), float(pars[1]), cfg["seed_idx"], T, IP_FIXED, rng)
    if scenario == "partial":
        inf_t = simulate_fixed_ip_curve(cfg["log_dist"], float(pars[0]), float(pars[1]), cfg["seed_idx"], T, IP_FIXED, rng)
        return apply_partial_observation(inf_t, cfg["seed_idx"], float(pars[2]), rng)
    return simulate_gamma_curve(cfg["log_dist"], float(pars[0]), float(pars[1]), float(pars[2]), cfg["seed_idx"], T, rng)


def select_epidemics(data):
    x, theta, obs_idx, conv = data["x"], data["theta"], data["obs_idx"], data["converged"]
    n_inf = np.array([int((x[o] > 0).sum()) for o in obs_idx])
    lo, hi = np.quantile(data["mcmc"], [0.025, 0.975], axis=1)
    mcmc_covers = np.all((theta[obs_idx] >= lo) & (theta[obs_idx] <= hi), axis=1)
    cand = np.where(conv & (n_inf >= MIN_OBS_INF) & mcmc_covers)[0]
    gnn = data["npe"]["GNN-NPE"][obs_idx[cand]]
    err = np.abs(np.median(gnn, axis=1) - theta[obs_idx[cand]])
    score = pd.DataFrame(err).rank(pct=True).to_numpy().mean(axis=1)
    rows = []
    for q in QUANTILES:
        j = int(np.argmin(np.abs(score - q)))
        k = int(cand[j])
        rows.append({
            "quantile": q, "local_mcmc_index": k, "obs_idx": int(obs_idx[k]),
            "gnn_error_rank_score": float(score[j]), "n_observed_infections": int(n_inf[k]),
            "n_candidates": int(len(cand)),
        })
    return pd.DataFrame(rows)


def run_selected(scenario, data, cfg, sel, out_dir, rng):
    T = data["T"]
    for _, row in sel.iterrows():
        k, o, q = int(row["local_mcmc_index"]), int(row["obs_idx"]), float(row["quantile"])
        x_obs = data["x"][o]
        arrays = {
            "obs_idx": np.int32(o),
            "quantile": np.float32(q),
            "theta": data["theta"][o].astype(np.float32),
            "param_names": np.array(data["param_names"], dtype=object),
            "observed_curve": inf_time_to_daily(x_obs[None, :], T)[0],
        }
        for m in METHODS:
            s = method_samples(data, m, k)
            draws = s[rng.choice(len(s), size=min(N_SEL_DRAWS, len(s)), replace=False)]
            sims = [simulate(scenario, p, cfg, T, rng) for p in draws]
            key = METHOD_KEYS[m]
            arrays[f"{key}_samples"] = s.astype(np.float32)
            arrays[f"{key}_curves"] = inf_time_to_daily(np.stack(sims), T).astype(np.float32)
        name = f"selected_q{int(round(100 * q)):02d}.npz"
        np.savez_compressed(out_dir / name, **arrays)
        print(f"  saved {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "gamma", "partial"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    scenario = args.scenario
    cfg = load_config()
    data = load_inputs(scenario, args.seed)
    out_dir = OUT_ROOT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SIM_SEED + {"full": 0, "gamma": 1, "partial": 2}[scenario])

    sel = select_epidemics(data)
    sel.insert(0, "scenario", scenario)
    sel["npe_seed"] = args.seed
    sel.to_csv(out_dir / "selection.csv", index=False)
    print(sel.to_string(index=False), flush=True)

    run_selected(scenario, data, cfg, sel, out_dir, rng)
    print(f"Saved outputs in {out_dir}")


if __name__ == "__main__":
    main()
