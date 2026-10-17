import argparse
import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn

from sim_02_make_datasets import (
    BASE_SEED as SIM02_BASE_SEED,
    generate_dataset,
    load_config as load_sim02_config,
    save_split,
)

from sim_03_tune_npe import (
    WeightedGNN,
    CNN1D,
    DEVICE,
    SEED,
    N_EVAL_POST,
    FIXED_HPARAMS,
    load_config,
    load_cnn,
    load_gnn,
    make_prior,
    set_seed,
    sample_valid_posterior,
)

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()
LC_ROOT = Path(os.environ.get("ILM_LC_ROOT", ARC_ROOT / "learning_curve")).expanduser().resolve()
RUNS_DIR = LC_ROOT / "runs"

SCENARIO = "full"
DEFAULT_N_VALUES = [1000, 2000, 5000, 10000, 20000, 30000, 50000]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
STRICT_FINAL_DIR = STRICT_ROOT / SCENARIO / "final"


def load_model_data(model, final_dir):
    if model == "gnn":
        return load_gnn(final_dir)
    return load_cnn(final_dir)


def load_best_config(model):
    path = STRICT_ROOT / SCENARIO / "best_hparams.json"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run sim_03_tune_npe.py first.")

    with open(path) as f:
        data = json.load(f)

    return data[model]["best_config"]


def build_emb(model, hps, T_len, positions, region_side):
    if model == "gnn":
        return WeightedGNN(
            positions,
            region_side,
            T_len,
            emb_dim=hps["emb_dim"],
            hidden=hps["gnn_hidden"],
            n_layers=hps["gnn_layers"],
            k=hps["gnn_k"],
            dropout=FIXED_HPARAMS["dropout"],
        ).to(DEVICE)

    return CNN1D(
        T_len,
        emb_dim=hps["emb_dim"],
        channels1=hps["channels1"],
        channels2=hps["channels2"],
        pool_out=FIXED_HPARAMS["pool_out"],
        dropout=FIXED_HPARAMS["dropout"],
    ).to(DEVICE)


def prepare_learning_curve_train_pool(target_n):
    lc_dir = LC_ROOT / SCENARIO / f"generated_final_train_{target_n}"
    train_path = lc_dir / "train.npz"
    test_path = lc_dir / "test.npz"

    if train_path.exists() and test_path.exists():
        dat = np.load(train_path, allow_pickle=True)
        if dat["theta"].shape[0] >= target_n:
            print(f"using existing learning-curve train pool: {lc_dir}")
            return lc_dir

    strict_train_path = STRICT_FINAL_DIR / "train.npz"
    strict_test_path = STRICT_FINAL_DIR / "test.npz"

    if not strict_train_path.exists():
        raise FileNotFoundError(f"Missing strict final train data: {strict_train_path}")

    if not strict_test_path.exists():
        raise FileNotFoundError(f"Missing strict final test data: {strict_test_path}")

    lc_dir.mkdir(parents=True, exist_ok=True)

    strict_train = np.load(strict_train_path, allow_pickle=True)
    th_base = strict_train["theta"].astype(np.float32)
    x_base = strict_train["x"]

    n_base = th_base.shape[0]

    if target_n <= n_base:
        th_combined = th_base[:target_n].copy()
        x_combined = x_base[:target_n].copy()

        base_stats_path = STRICT_FINAL_DIR / "train_stats.csv"
        if base_stats_path.exists():
            stats_combined = pd.read_csv(base_stats_path).iloc[:target_n].copy()
        else:
            stats_combined = pd.DataFrame()

        elapsed = 0.0
    else:
        n_extra = target_n - n_base
        print(f"generating extra full train simulations for learning curve only: {n_extra}")
        print(f"output directory: {lc_dir}")

        cfg02 = load_sim02_config()
        extra_seed = SIM02_BASE_SEED + 900000 + target_n

        t0 = time.perf_counter()
        th_extra, x_extra, stats_extra = generate_dataset(
            SCENARIO,
            n_extra,
            cfg02,
            extra_seed,
        )
        elapsed = time.perf_counter() - t0

        th_combined = np.concatenate([th_base, th_extra.astype(np.float32)], axis=0)
        x_combined = np.concatenate([x_base, x_extra], axis=0)

        base_stats_path = STRICT_FINAL_DIR / "train_stats.csv"

        if base_stats_path.exists():
            stats_base = pd.read_csv(base_stats_path)
            stats_combined = pd.concat([stats_base, stats_extra], ignore_index=True)
        else:
            stats_combined = stats_extra.copy()

    save_split(
        lc_dir,
        SCENARIO,
        "train",
        th_combined,
        x_combined,
        stats_combined,
        elapsed,
    )

    for name in ["test.npz", "test_stats.csv", "test_meta.json"]:
        src = STRICT_FINAL_DIR / name
        dst = lc_dir / name

        if src.exists():
            shutil.copy2(src, dst)

    meta_path = lc_dir / "learning_curve_source.json"

    with open(meta_path, "w") as f:
        json.dump(
            {
                "scenario": SCENARIO,
                "target_train_n": int(target_n),
                "strict_train_n": int(n_base),
                "extra_train_n": int(max(0, target_n - n_base)),
                "note": "Extra training simulations are generated only for learning-curve analysis. Strict final outputs are not modified.",
            },
            f,
            indent=2,
        )

    return lc_dir


def choose_data_dir_for_n_values(N_values):
    train_npz = STRICT_FINAL_DIR / "train.npz"

    if not train_npz.exists():
        raise FileNotFoundError(f"Missing strict final train data: {train_npz}")

    dat = np.load(train_npz, allow_pickle=True)
    n_strict = dat["theta"].shape[0]

    required_train_n = max(N_values)

    if required_train_n <= n_strict:
        return STRICT_FINAL_DIR, n_strict

    lc_dir = prepare_learning_curve_train_pool(required_train_n)

    dat_lc = np.load(lc_dir / "train.npz", allow_pickle=True)
    return lc_dir, dat_lc["theta"].shape[0]


def build_nested_train_index(n_train_total, n_strict_base):
    rng = np.random.default_rng(SEED)

    if n_train_total <= n_strict_base:
        return rng.permutation(n_train_total)

    strict_part = rng.permutation(n_strict_base)
    extra_part = n_strict_base + rng.permutation(n_train_total - n_strict_base)

    return np.concatenate([strict_part, extra_part])


def train_and_eval(
    th_tr_sub,
    x_tr_sub,
    th_te,
    x_te,
    prior,
    emb,
    hps,
    lut_betas,
    lut_seed_rowsums,
    log_dist,
    seed_idx,
    seed,
):
    set_seed(seed)

    de_fn = posterior_nn(
        model=FIXED_HPARAMS["flow_model"],
        embedding_net=emb,
        hidden_features=hps["hidden_features"],
        num_transforms=hps["num_transforms"],
    )

    inf = SNPE(prior=prior, density_estimator=de_fn, device=DEVICE)

    inf.append_simulations(
        torch.tensor(th_tr_sub, dtype=torch.float32, device=DEVICE),
        torch.tensor(x_tr_sub, dtype=torch.float32, device=DEVICE),
    )

    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        de = inf.train(
            max_num_epochs=FIXED_HPARAMS["max_num_epochs"],
            training_batch_size=FIXED_HPARAMS["batch_size"],
            learning_rate=hps["learning_rate"],
            validation_fraction=FIXED_HPARAMS["validation_fraction"],
            stop_after_epochs=FIXED_HPARAMS["stop_after_epochs"],
            show_train_summary=False,
        )

    train_sec = time.perf_counter() - t0
    n_epochs = 0
    best_epoch = -1
    best_val_lp = float("nan")

    try:
        val_log_probs = inf.summary.get("validation_log_probs", [])
        if val_log_probs:
            n_epochs = len(val_log_probs)
            best_epoch = int(np.argmax(val_log_probs))
            best_val_lp = float(np.max(val_log_probs))
    except Exception:
        pass

    posterior = inf.build_posterior(de)

    set_seed(seed)
    n, p = th_te.shape
    samples = np.empty((n, N_EVAL_POST, p), dtype=np.float32)
    valid_obs = 0

    for i in range(n):
        x_obs = torch.tensor(x_te[i:i + 1], dtype=torch.float32, device=DEVICE)

        draw = sample_valid_posterior(
            posterior,
            x_obs,
            SCENARIO,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
            N_EVAL_POST,
        )

        if draw is None:
            samples[i] = np.nan
        else:
            samples[i] = draw
            valid_obs += 1

    ok = np.isfinite(samples).all(axis=(1, 2))

    if not ok.any():
        raise RuntimeError("No test observation produced enough valid posterior samples.")

    s_ok = samples[ok]
    th_ok = th_te[ok]

    medians = np.median(s_ok, axis=1)
    abs_err = np.abs(medians - th_ok)

    mae_per_param = abs_err.mean(axis=0)
    mae_overall = float(abs_err.mean())

    qlo = np.percentile(s_ok, 2.5, axis=1)
    qhi = np.percentile(s_ok, 97.5, axis=1)
    cover95_per_param = ((th_ok >= qlo) & (th_ok <= qhi)).mean(axis=0)
    width95_per_param = (qhi - qlo).mean(axis=0)

    return {
        "mae_overall": mae_overall,
        "mae_per_param": mae_per_param.tolist(),
        "cover95_per_param": cover95_per_param.tolist(),
        "width95_per_param": width95_per_param.tolist(),
        "train_sec": float(train_sec),
        "n_epochs": int(n_epochs),
        "best_epoch": int(best_epoch),
        "best_val_lp": best_val_lp,
        "valid_obs": int(valid_obs),
        "valid_rate": float(valid_obs / n),
    }


def run_path(model, N, rep_seed):
    return RUNS_DIR / f"{model}_N{N:05d}_seed{rep_seed}.json"


def run_learning_curve(model, N_values, rep_seeds, cfg):
    print(f"\n{'=' * 60}")
    print(f"scenario={SCENARIO} model={model} seeds={rep_seeds}")
    print(f"{'=' * 60}")

    data_dir, n_train_total = choose_data_dir_for_n_values(N_values)

    prior = make_prior(
        SCENARIO,
        cfg["lut_betas"],
        cfg["lut_seed_rowsums"],
        cfg["log_dist"],
        cfg["seed_idx"],
    )

    hps_best = load_best_config(model)
    hps = {**FIXED_HPARAMS, **hps_best}

    print(f"hparams: {hps_best}")
    print(f"data directory: {data_dir}")

    th_tr_full, x_tr_full, th_te, x_te, T_len = load_model_data(model, data_dir)

    n_train_avail = th_tr_full.shape[0]
    n_test = th_te.shape[0]

    if n_train_avail < n_train_total:
        raise ValueError(f"Loaded train pool has {n_train_avail}, expected at least {n_train_total}.")

    strict_train = np.load(STRICT_FINAL_DIR / "train.npz", allow_pickle=True)
    n_strict_base = strict_train["theta"].shape[0]

    master_train_idx = build_nested_train_index(n_train_avail, n_strict_base)

    print(f"available train pool: {n_train_avail}")
    print(f"strict base train pool: {n_strict_base}")
    print(f"fixed test set: {n_test}")

    todo = [(rep, N) for rep in rep_seeds for N in N_values if not run_path(model, N, rep).exists()]
    print(f"runs to do: {len(todo)}  (already done: {len(rep_seeds) * len(N_values) - len(todo)})")

    grand_t0 = time.perf_counter()

    for run_i, (rep, N) in enumerate(todo, start=1):
        if N > n_train_avail:
            raise ValueError(f"N={N} > available train pool {n_train_avail}")

        train_seed = rep + N

        print(f"\n[{SCENARIO}/{model}] [{run_i}/{len(todo)}] N={N} rep_seed={rep} train_seed={train_seed}")

        t0 = time.perf_counter()

        train_idx = master_train_idx[:N]

        th_tr_sub = th_tr_full[train_idx].copy()
        x_tr_sub = x_tr_full[train_idx].copy()

        set_seed(train_seed)
        emb = build_emb(model, hps, T_len, cfg["positions"], cfg["region_side"])

        res = train_and_eval(
            th_tr_sub,
            x_tr_sub,
            th_te,
            x_te,
            prior,
            emb,
            hps,
            cfg["lut_betas"],
            cfg["lut_seed_rowsums"],
            cfg["log_dist"],
            cfg["seed_idx"],
            train_seed,
        )

        wall = time.perf_counter() - t0

        row = {
            "scenario": SCENARIO,
            "model": model,
            "N": int(N),
            "rep_seed": int(rep),
            "train_seed": int(train_seed),
            "n_test": int(n_test),
            "mae_overall": res["mae_overall"],
            "train_sec": res["train_sec"],
            "wall_sec": float(wall),
            "n_epochs": res["n_epochs"],
            "best_epoch": res["best_epoch"],
            "best_val_lp": res["best_val_lp"],
            "valid_obs": res["valid_obs"],
            "valid_rate": res["valid_rate"],
            "mae_alpha": res["mae_per_param"][0],
            "mae_beta": res["mae_per_param"][1],
            "cover95_alpha": res["cover95_per_param"][0],
            "cover95_beta": res["cover95_per_param"][1],
            "width95_alpha": res["width95_per_param"][0],
            "width95_beta": res["width95_per_param"][1],
        }

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = run_path(model, N, rep).with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(row, f, indent=2)
        os.replace(tmp, run_path(model, N, rep))

        print(
            f"mae_alpha={row['mae_alpha']:.4f} mae_beta={row['mae_beta']:.4f} "
            f"cov_alpha={row['cover95_alpha']:.3f} cov_beta={row['cover95_beta']:.3f} "
            f"epochs={res['n_epochs']}(best={res['best_epoch']}) "
            f"valid={res['valid_obs']}/{n_test} train={res['train_sec']:.0f}s"
        )

    elapsed_min = (time.perf_counter() - grand_t0) / 60
    print(f"\n[{SCENARIO}/{model}] runs done in {elapsed_min:.1f} min")


def aggregate():
    files = sorted(RUNS_DIR.glob("*_N*_seed*.json"))

    if not files:
        print("No learning-curve runs found to aggregate.")
        return

    rows = []
    for p in files:
        with open(p) as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows).sort_values(["model", "N", "rep_seed"]).reset_index(drop=True)

    all_path = LC_ROOT / f"learning_curve_{SCENARIO}_all.csv"
    df.to_csv(all_path, index=False)

    metrics = [
        "mae_alpha",
        "mae_beta",
        "cover95_alpha",
        "cover95_beta",
        "width95_alpha",
        "width95_beta",
        "train_sec",
        "n_epochs",
        "valid_rate",
    ]

    summary = df.groupby(["model", "N"])[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in summary.columns.to_flat_index()
    ]

    summary_path = LC_ROOT / f"learning_curve_{SCENARIO}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"saved: {all_path}")
    print(f"saved: {summary_path}")

    for _, r in summary.iterrows():
        print(
            f"{r['model']:>4} N={int(r['N']):>6}  "
            f"MAE alpha={r['mae_alpha_mean']:.4f}±{r['mae_alpha_std']:.4f}  "
            f"MAE beta={r['mae_beta_mean']:.4f}±{r['mae_beta_std']:.4f}  "
            f"cov beta={r['cover95_beta_mean']:.3f}  (n={int(r['mae_alpha_count'])})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["gnn", "cnn", "all"])
    parser.add_argument("--n_values", default=",".join(str(x) for x in DEFAULT_N_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--aggregate-only", action="store_true")

    args = parser.parse_args()

    LC_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate()
        return

    N_values = [int(x) for x in args.n_values.split(",")]
    models = ["gnn", "cnn"] if args.model == "all" else [args.model]

    cfg = load_config()

    print(f"device={DEVICE}")
    print(f"scenario={SCENARIO}")
    print(f"N_values={N_values}")
    print(f"seeds={args.seeds}")
    print(f"models={models}")
    print(f"strict final dir={STRICT_FINAL_DIR}")
    print(f"learning curve root={LC_ROOT}")

    grand_t0 = time.perf_counter()

    for model in models:
        run_learning_curve(model, N_values, args.seeds, cfg)

    grand_min = (time.perf_counter() - grand_t0) / 60

    print(f"\n{'=' * 60}")
    print(f"All runs done. Total wall time: {grand_min:.1f} min")
    print(f"{'=' * 60}")

    aggregate()


if __name__ == "__main__":
    main()
