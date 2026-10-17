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

IP_FIXED = 3
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
N_PRED_DRAWS = 100
PRED_SEED = 1234


def load_config():
    dat = np.load(CONFIG_ROOT / "ilm_config.npz", allow_pickle=True)
    return {
        "N": int(dat["N"]),
        "region_side": float(dat["region_side"]),
        "log_dist": dat["log_dist"].astype(np.float64),
        "seed_idx": dat["seed_idx"].astype(np.int32),
    }


def inf_time_to_daily(x, T_len):
    c = np.zeros((x.shape[0], T_len), dtype=np.float32)

    for t in range(1, T_len + 1):
        c[:, t - 1] = (x == t).sum(axis=1)

    return c


def load_test_data(final_dir):
    te = np.load(final_dir / "test.npz", allow_pickle=True)
    return {
        "theta": te["theta"].astype(np.float32),
        "x": te["x"].astype(np.int32),
        "T": int(te["T"]),
        "param_names": tuple(te["param_names"].tolist()),
    }


def load_npe_outputs(final_dir, model_tag):
    npz = np.load(final_dir / f"{model_tag}_posterior_samples.npz", allow_pickle=True)

    with open(final_dir / f"{model_tag}_run_meta.json") as f:
        meta = json.load(f)

    return npz["samples"].astype(np.float32), meta


def compute_param_metrics(samples, theta_true, param_names):
    rows = []

    for i in range(theta_true.shape[0]):
        for j, pname in enumerate(param_names):
            s = samples[i, :, j]
            tv = float(theta_true[i, j])
            pmed = float(np.median(s))

            row = {
                "obs_id": int(i),
                "parameter": pname,
                "true": tv,
                "post_median": pmed,
                "post_mean": float(np.mean(s)),
                "post_sd": float(np.std(s, ddof=1)),
                "abs_error": abs(pmed - tv),
                "sq_error": (pmed - tv) ** 2,
                "bias": pmed - tv,
                "rel_error": abs(pmed - tv) / max(abs(tv), 1e-12),
            }

            for lv in COVERAGE_LEVELS:
                lo = float(np.quantile(s, (1 - lv) / 2))
                hi = float(np.quantile(s, 1 - (1 - lv) / 2))
                row[f"coverage_{int(lv * 100)}"] = float(lo <= tv <= hi)

                if lv == 0.95:
                    row["ci95_width"] = hi - lo

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_param_metrics(df):
    out = []

    for pname, g in df.groupby("parameter"):
        row = {
            "parameter": pname,
            "mae": float(g["abs_error"].mean()),
            "rmse": float(np.sqrt(g["sq_error"].mean())),
            "mean_bias": float(g["bias"].mean()),
            "mean_rel_error": float(g["rel_error"].mean()),
            "mean_ci95_width": float(g["ci95_width"].mean()),
        }

        for lv in COVERAGE_LEVELS:
            row[f"coverage_{int(lv * 100)}"] = float(g[f"coverage_{int(lv * 100)}"].mean())

        out.append(row)

    return pd.DataFrame(out)


def simulate_fixed_ip_curve(log_dist, alpha, beta, seed_idx, T_len, ip_fixed, rng):
    n = log_dist.shape[0]
    W = np.exp(-beta * log_dist)
    np.fill_diagonal(W, 0.0)

    infection_time = np.full(n, -1, dtype=np.int32)
    infection_time[seed_idx] = 0

    for t in range(T_len):
        inf_mask = (infection_time >= 0) & ((t - infection_time) < ip_fixed)
        susc_mask = infection_time < 0

        if not inf_mask.any():
            break

        pressure = alpha * (W @ inf_mask.astype(np.float64))
        p = np.clip(1.0 - np.exp(-pressure), 0.0, 1.0)
        p[~susc_mask] = 0.0

        new_inf = (rng.random(n) < p) & susc_mask
        infection_time[new_inf] = t + 1

    return infection_time


def apply_partial_observation(infection_time, seed_idx, rho, rng):
    n = len(infection_time)
    infected_idx = np.where(infection_time >= 0)[0]
    obs_mask = np.zeros(n, dtype=bool)

    if len(infected_idx) > 0:
        kept = rng.random(len(infected_idx)) < rho
        obs_mask[infected_idx[kept]] = True

    obs_mask[seed_idx] = True

    x_obs = infection_time.copy()
    x_obs[(infection_time >= 0) & (~obs_mask)] = -1

    return x_obs


def simulate_gamma_curve(log_dist, alpha, beta, gamma, seed_idx, T_len, rng):
    n = log_dist.shape[0]
    p_ir = 1.0 - math.exp(-gamma)
    W = np.exp(-beta * log_dist)
    np.fill_diagonal(W, 0.0)

    infection_time = np.full(n, -1, dtype=np.int32)
    recovered = np.zeros(n, dtype=bool)
    infection_time[seed_idx] = 0

    for t in range(T_len):
        inf_mask = (infection_time >= 0) & (~recovered)
        susc_mask = infection_time < 0

        if not inf_mask.any():
            break

        pressure = alpha * (W @ inf_mask.astype(np.float64))
        p = np.clip(1.0 - np.exp(-pressure), 0.0, 1.0)
        p[~susc_mask] = 0.0

        new_inf = (rng.random(n) < p) & susc_mask
        infection_time[new_inf] = t + 1

        inf_mask_after_transmission = (infection_time >= 0) & (~recovered)
        recovered[inf_mask_after_transmission & (rng.random(n) < p_ir)] = True

    return infection_time


def compute_predictive_curves(scenario, samples, log_dist, seed_idx, T_len, n_draws, seed):
    n_obs, n_post, _ = samples.shape
    n_draws = min(n_draws, n_post)
    curves = np.empty((n_obs, n_draws, T_len), dtype=np.float32)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    for i in range(n_obs):
        draw_idx = rng.choice(n_post, size=n_draws, replace=False)

        for k, di in enumerate(draw_idx):
            pars = samples[i, di]

            if scenario == "full":
                x = simulate_fixed_ip_curve(
                    log_dist,
                    float(pars[0]),
                    float(pars[1]),
                    seed_idx,
                    T_len,
                    IP_FIXED,
                    rng,
                )
            elif scenario == "partial":
                inf_t = simulate_fixed_ip_curve(
                    log_dist,
                    float(pars[0]),
                    float(pars[1]),
                    seed_idx,
                    T_len,
                    IP_FIXED,
                    rng,
                )
                rho = float(pars[2])

                if not (0.0 < rho < 1.0):
                    raise ValueError(f"Invalid rho: {rho}")

                x = apply_partial_observation(inf_t, seed_idx, rho, rng)
            else:
                gamma = float(pars[2])

                if gamma <= 0.0:
                    raise ValueError(f"Invalid gamma: {gamma}")

                x = simulate_gamma_curve(
                    log_dist,
                    float(pars[0]),
                    float(pars[1]),
                    gamma,
                    seed_idx,
                    T_len,
                    rng,
                )

            curves[i, k] = inf_time_to_daily(x[None, :], T_len)[0]

        if (i + 1) % 100 == 0 or i + 1 == n_obs:
            print(f"  PPC {i + 1}/{n_obs}  {time.perf_counter() - t0:.0f}s", flush=True)

    return curves


def extract_runtime_npe(meta):
    timing = meta.get("timing", {})
    return {
        "train_sec": timing.get("train_sec", np.nan),
        "infer_sec": timing.get("infer_sec", np.nan),
        "ms_per_obs": timing.get("ms_per_obs", np.nan),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "partial", "gamma"])
    args = parser.parse_args()

    scenario = args.scenario
    cfg = load_config()

    scenario_dir = STRICT_ROOT / scenario
    final_dir = scenario_dir / "final"
    eval_dir = scenario_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test = load_test_data(final_dir)
    theta_test = test["theta"]
    x_test = test["x"]
    T_len = test["T"]
    param_names = test["param_names"]
    observed_curves = inf_time_to_daily(x_test, T_len)

    print(f"scenario={scenario}  n_test={theta_test.shape[0]}  T={T_len}")
    print(f"params: {param_names}")

    samples_cnn, cnn_meta = load_npe_outputs(final_dir, "cnn")
    samples_gnn, gnn_meta = load_npe_outputs(final_dir, "gnn")

    if samples_cnn.shape[0] != samples_gnn.shape[0]:
        raise ValueError("CNN and GNN posterior sample observation counts do not match.")

    if samples_cnn.shape[1] != samples_gnn.shape[1]:
        raise ValueError("CNN and GNN posterior sample counts do not match.")

    if samples_cnn.shape[2] != samples_gnn.shape[2]:
        raise ValueError("CNN and GNN posterior sample parameter dimensions do not match.")

    if samples_cnn.shape[0] != theta_test.shape[0]:
        raise ValueError("Posterior sample observation count does not match theta_test.")

    if samples_cnn.shape[2] != theta_test.shape[1]:
        raise ValueError("Posterior sample parameter dimension does not match theta_test.")

    print(f"CNN samples: {samples_cnn.shape}  GNN samples: {samples_gnn.shape}")

    df_cnn = compute_param_metrics(samples_cnn, theta_test, param_names)
    df_gnn = compute_param_metrics(samples_gnn, theta_test, param_names)
    sm_cnn = summarize_param_metrics(df_cnn)
    sm_gnn = summarize_param_metrics(df_gnn)

    df_cnn.to_csv(eval_dir / "param_level_cnn.csv", index=False)
    df_gnn.to_csv(eval_dir / "param_level_gnn.csv", index=False)
    sm_cnn.to_csv(eval_dir / "param_summary_cnn.csv", index=False)
    sm_gnn.to_csv(eval_dir / "param_summary_gnn.csv", index=False)

    print("Computing PPC curves (CNN)...")
    curves_cnn = compute_predictive_curves(
        scenario,
        samples_cnn,
        cfg["log_dist"],
        cfg["seed_idx"],
        T_len,
        N_PRED_DRAWS,
        PRED_SEED,
    )

    print("Computing PPC curves (GNN)...")
    curves_gnn = compute_predictive_curves(
        scenario,
        samples_gnn,
        cfg["log_dist"],
        cfg["seed_idx"],
        T_len,
        N_PRED_DRAWS,
        PRED_SEED + 1,
    )

    np.savez_compressed(
        eval_dir / "ppc_curves_npe_full.npz",
        observed_curves=observed_curves.astype(np.float32),
        cnn_curves=curves_cnn.astype(np.float32),
        gnn_curves=curves_gnn.astype(np.float32),
    )

    summary_rows = []

    for model_name, summary_df, meta in [
        ("CNN-NPE", sm_cnn, cnn_meta),
        ("GNN-NPE", sm_gnn, gnn_meta),
    ]:
        rt = extract_runtime_npe(meta)

        for _, row in summary_df.iterrows():
            summary_rows.append(
                {
                    "scenario": scenario,
                    "scope": "full_test",
                    "model": model_name,
                    "parameter": row["parameter"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mean_bias": row["mean_bias"],
                    "mean_rel_error": row["mean_rel_error"],
                    "mean_ci95_width": row["mean_ci95_width"],
                    "coverage_95": row["coverage_95"],
                    "train_sec": rt["train_sec"],
                    "infer_sec": rt["infer_sec"],
                    "ms_per_obs": rt["ms_per_obs"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "evaluation_summary_npe_full.csv", index=False)

    print(f"\nSaved evaluation outputs to {eval_dir}")
    print("\nCNN summary:")
    print(sm_cnn[["parameter", "mae", "coverage_95"]].to_string(index=False))
    print("\nGNN summary:")
    print(sm_gnn[["parameter", "mae", "coverage_95"]].to_string(index=False))


if __name__ == "__main__":
    main()