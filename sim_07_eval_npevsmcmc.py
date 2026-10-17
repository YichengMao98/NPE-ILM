import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sim_06_eval_cnnvsgnn import (
    N_PRED_DRAWS,
    PRED_SEED,
    load_config,
    load_test_data,
    load_npe_outputs,
    inf_time_to_daily,
    compute_param_metrics,
    summarize_param_metrics,
    compute_predictive_curves,
    extract_runtime_npe,
)

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()

DIAG_ROOT = ARC_ROOT / "mcmc_diagnostics"


def load_mcmc_outputs(mcmc_dir):
    sample_path = mcmc_dir / "mcmc_samples.npz"
    theta_path = mcmc_dir / "mcmc_theta_test.npy"
    obs_idx_path = mcmc_dir / "mcmc_obs_idx.npy"
    meta_path = mcmc_dir / "mcmc_run_meta.json"

    if not sample_path.exists():
        return None

    if not theta_path.exists():
        raise FileNotFoundError(f"Missing MCMC theta file: {theta_path}")

    if not obs_idx_path.exists():
        raise FileNotFoundError(f"Missing MCMC obs_idx file: {obs_idx_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing MCMC meta file: {meta_path}")

    npz = np.load(sample_path, allow_pickle=True)
    theta = np.load(theta_path).astype(np.float32)
    obs_idx = np.load(obs_idx_path).astype(np.int32)

    with open(meta_path) as f:
        meta = json.load(f)

    raw_samples = npz["samples"].astype(np.float32)

    if raw_samples.ndim != 4:
        raise ValueError(
            f"MCMC samples must have shape (n_obs, n_chains, n_keep, n_params), got {raw_samples.shape}"
        )

    n_obs, n_chains, n_keep, n_params = raw_samples.shape

    if theta.shape != (n_obs, n_params):
        raise ValueError(f"MCMC theta shape {theta.shape} does not match samples shape {raw_samples.shape}")

    if obs_idx.shape[0] != n_obs:
        raise ValueError(f"MCMC obs_idx length {obs_idx.shape[0]} does not match n_obs {n_obs}")

    return raw_samples, theta, obs_idx, meta


def flatten_mcmc_samples(raw_samples):
    n_obs, n_chains, n_keep, n_params = raw_samples.shape
    return raw_samples.reshape(n_obs, n_chains * n_keep, n_params)


def load_convergence(scenario, obs_idx, eval_dir):
    path = DIAG_ROOT / f"diagnostics_by_obs_{scenario}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MCMC diagnostics: {path}\nRun sim_09_mcmc_diagnostics.py first.")
    diag = pd.read_csv(path).sort_values("local_mcmc_index").reset_index(drop=True)
    if len(diag) != len(obs_idx) or not np.array_equal(diag["obs_idx"].to_numpy(), np.asarray(obs_idx)):
        raise ValueError(f"MCMC diagnostics in {path} do not match the MCMC observations.")
    diag.to_csv(eval_dir / "mcmc_convergence_by_obs.csv", index=False)
    with open(DIAG_ROOT / "diagnostics_settings.json") as f:
        settings = json.load(f)
    converged = diag["converged"].to_numpy(bool)
    with open(eval_dir / "mcmc_convergence_summary.json", "w") as f:
        json.dump({
            "scenario": scenario,
            "criterion": settings,
            "n_mcmc_total": int(len(diag)),
            "n_mcmc_converged": int(converged.sum()),
            "pct_mcmc_converged": float(100.0 * converged.mean()),
        }, f, indent=2)
    np.save(eval_dir / "mcmc_converged_obs_idx.npy", np.asarray(obs_idx)[converged].astype(np.int32))
    return converged


def extract_runtime_mcmc(meta):
    timing = meta.get("timing", {})
    mean_cpu_per_obs = timing.get("mean_cpu_per_obs", timing.get("mean_cpu_per_obs_sec", np.nan))

    return {
        "train_sec": np.nan,
        "infer_sec": timing.get("total_wall_sec", np.nan),
        "ms_per_obs": np.nan,
        "cpu_ms_per_obs": float(mean_cpu_per_obs) * 1000 if np.isfinite(mean_cpu_per_obs) else np.nan,
    }


def check_alignment(samples_cnn, samples_gnn, raw_samples_mcmc, theta_mcmc, obs_idx, test):
    if theta_mcmc.shape[0] != len(obs_idx):
        raise ValueError("theta_mcmc and obs_idx length do not match.")

    if obs_idx.min() < 0:
        raise ValueError("obs_idx contains negative indices.")

    if obs_idx.max() >= test["theta"].shape[0]:
        raise ValueError("obs_idx exceeds test theta size.")

    if obs_idx.max() >= samples_cnn.shape[0] or obs_idx.max() >= samples_gnn.shape[0]:
        raise ValueError("obs_idx exceeds NPE posterior sample size.")

    if samples_cnn.shape[0] != samples_gnn.shape[0]:
        raise ValueError("CNN and GNN NPE observation counts do not match.")

    if samples_cnn.shape[2] != samples_gnn.shape[2]:
        raise ValueError("CNN and GNN NPE parameter dimensions do not match.")

    if samples_cnn.shape[2] != theta_mcmc.shape[1]:
        raise ValueError("CNN NPE parameter dimension does not match MCMC theta.")

    if samples_gnn.shape[2] != theta_mcmc.shape[1]:
        raise ValueError("GNN NPE parameter dimension does not match MCMC theta.")

    if raw_samples_mcmc.shape[3] != theta_mcmc.shape[1]:
        raise ValueError("MCMC sample parameter dimension does not match MCMC theta.")

    theta_from_full_test = test["theta"][obs_idx].astype(np.float32)

    if not np.allclose(theta_from_full_test, theta_mcmc, rtol=1e-5, atol=1e-6):
        raise ValueError("MCMC theta does not match full test theta indexed by obs_idx.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "partial", "gamma"])
    args = parser.parse_args()

    scenario = args.scenario
    cfg = load_config()

    scenario_dir = STRICT_ROOT / scenario
    final_dir = scenario_dir / "final"
    mcmc_dir = scenario_dir / "mcmc"
    eval_dir = scenario_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    mcmc_loaded = load_mcmc_outputs(mcmc_dir)

    if mcmc_loaded is None:
        raise FileNotFoundError(
            f"MCMC outputs not found in {mcmc_dir}.\n"
            f"Run sim_05_mcmc_da.py for each observation and then with --aggregate."
        )

    raw_samples_mcmc, theta_mcmc_all, obs_idx_all, mcmc_meta = mcmc_loaded

    print(f"MCMC raw samples: {raw_samples_mcmc.shape}  obs_idx: {obs_idx_all.tolist()}")

    test = load_test_data(final_dir)
    T_len = test["T"]
    param_names = test["param_names"]
    observed_curves = inf_time_to_daily(test["x"], T_len)

    if len(param_names) != theta_mcmc_all.shape[1]:
        raise ValueError("param_names length does not match theta_mcmc dimension.")

    samples_cnn, cnn_meta = load_npe_outputs(final_dir, "cnn")
    samples_gnn, gnn_meta = load_npe_outputs(final_dir, "gnn")

    check_alignment(samples_cnn, samples_gnn, raw_samples_mcmc, theta_mcmc_all, obs_idx_all, test)

    converged_mask = load_convergence(scenario, obs_idx_all, eval_dir)

    n_total = int(len(obs_idx_all))
    n_converged = int(converged_mask.sum())

    print(f"\nMCMC convergence: {n_converged}/{n_total} observations pass the sim_09 criterion")

    if n_converged == 0:
        raise ValueError("No MCMC observations satisfy the convergence criterion.")

    raw_samples_mcmc = raw_samples_mcmc[converged_mask]
    theta_mcmc = theta_mcmc_all[converged_mask]
    obs_idx = obs_idx_all[converged_mask]
    samples_mcmc = flatten_mcmc_samples(raw_samples_mcmc)

    samples_cnn_sub = samples_cnn[obs_idx]
    samples_gnn_sub = samples_gnn[obs_idx]
    curves_obs_sub = observed_curves[obs_idx]

    print(f"\nUsing converged MCMC subset: n={len(obs_idx)}")
    print(f"NPE subset shapes: CNN={samples_cnn_sub.shape}  GNN={samples_gnn_sub.shape}")
    print(
        f"Sample counts (per obs): NPE-CNN={samples_cnn_sub.shape[1]}  "
        f"NPE-GNN={samples_gnn_sub.shape[1]}  MCMC={samples_mcmc.shape[1]}"
    )

    df_cnn_sub = compute_param_metrics(samples_cnn_sub, theta_mcmc, param_names)
    df_gnn_sub = compute_param_metrics(samples_gnn_sub, theta_mcmc, param_names)
    df_mcmc = compute_param_metrics(samples_mcmc, theta_mcmc, param_names)

    sm_cnn_sub = summarize_param_metrics(df_cnn_sub)
    sm_gnn_sub = summarize_param_metrics(df_gnn_sub)
    sm_mcmc = summarize_param_metrics(df_mcmc)

    df_cnn_sub.to_csv(eval_dir / "param_level_cnn_mcmc_subset.csv", index=False)
    df_gnn_sub.to_csv(eval_dir / "param_level_gnn_mcmc_subset.csv", index=False)
    df_mcmc.to_csv(eval_dir / "param_level_mcmc.csv", index=False)
    sm_cnn_sub.to_csv(eval_dir / "param_summary_cnn_mcmc_subset.csv", index=False)
    sm_gnn_sub.to_csv(eval_dir / "param_summary_gnn_mcmc_subset.csv", index=False)
    sm_mcmc.to_csv(eval_dir / "param_summary_mcmc.csv", index=False)

    print("Computing PPC curves (CNN subset)...")
    curves_cnn_sub = compute_predictive_curves(
        scenario,
        samples_cnn_sub,
        cfg["log_dist"],
        cfg["seed_idx"],
        T_len,
        N_PRED_DRAWS,
        PRED_SEED + 2,
    )

    print("Computing PPC curves (GNN subset)...")
    curves_gnn_sub = compute_predictive_curves(
        scenario,
        samples_gnn_sub,
        cfg["log_dist"],
        cfg["seed_idx"],
        T_len,
        N_PRED_DRAWS,
        PRED_SEED + 3,
    )

    print("Computing PPC curves (MCMC)...")
    curves_mcmc = compute_predictive_curves(
        scenario,
        samples_mcmc,
        cfg["log_dist"],
        cfg["seed_idx"],
        T_len,
        N_PRED_DRAWS,
        PRED_SEED + 4,
    )

    np.savez_compressed(
        eval_dir / "ppc_curves_mcmc_subset.npz",
        obs_idx=obs_idx.astype(np.int32),
        observed_curves=curves_obs_sub.astype(np.float32),
        cnn_curves=curves_cnn_sub.astype(np.float32),
        gnn_curves=curves_gnn_sub.astype(np.float32),
        mcmc_curves=curves_mcmc.astype(np.float32),
    )

    summary_rows = []

    for model_name, summary_df, meta, rt_fn in [
        ("CNN-NPE", sm_cnn_sub, cnn_meta, extract_runtime_npe),
        ("GNN-NPE", sm_gnn_sub, gnn_meta, extract_runtime_npe),
        ("MCMC", sm_mcmc, mcmc_meta, extract_runtime_mcmc),
    ]:
        rt = rt_fn(meta)

        for _, row in summary_df.iterrows():
            summary_rows.append(
                {
                    "scenario": scenario,
                    "scope": "mcmc_converged_subset",
                    "n_mcmc_total": n_total,
                    "n_mcmc_converged": n_converged,
                    "model": model_name,
                    "parameter": row["parameter"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mean_bias": row["mean_bias"],
                    "mean_rel_error": row["mean_rel_error"],
                    "mean_ci95_width": row["mean_ci95_width"],
                    "coverage_95": row["coverage_95"],
                    "train_sec": rt.get("train_sec", np.nan),
                    "infer_sec": rt.get("infer_sec", np.nan),
                    "ms_per_obs": rt.get("ms_per_obs", np.nan),
                    "cpu_ms_per_obs": rt.get("cpu_ms_per_obs", np.nan),
                }
            )

    summary_df_out = pd.DataFrame(summary_rows)
    summary_df_out.to_csv(eval_dir / "evaluation_summary_npe_vs_mcmc.csv", index=False)

    print(f"\nSaved 3-way comparison outputs to {eval_dir}")
    print("\n=== MAE / Coverage 95% on converged MCMC subset ===")

    for label, sm in [("CNN-NPE", sm_cnn_sub), ("GNN-NPE", sm_gnn_sub), ("MCMC", sm_mcmc)]:
        print(f"\n{label}:")
        print(sm[["parameter", "mae", "mean_rel_error", "coverage_95"]].to_string(index=False))


if __name__ == "__main__":
    main()