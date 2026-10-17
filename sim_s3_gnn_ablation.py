import argparse
import csv
import json
import time

import numpy as np
import torch

from sim_03_tune_npe import (
    GNN,
    DEVICE,
    FIXED_HPARAMS,
    load_config,
    load_gnn,
    make_prior,
    set_seed,
)
from sim_04_train_infer import (
    ARC_ROOT,
    STRICT_ROOT,
    SEEDS_ROOT,
    PARAM_NAMES,
    count_params,
    density_estimator,
    run_inference,
    train_npe,
)

OUT_ROOT = ARC_ROOT / "ablation_gnn"
SCENARIOS = ("full", "gamma", "partial")
MAIN_VARIANT = "V1_knn_weighted"
TRAINED_VARIANTS = ("V0_knn_mean",)
VARIANTS = (MAIN_VARIANT, *TRAINED_VARIANTS)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def gnn_hparams(saved, seed):
    best = saved["gnn"]["best_config"]
    return {
        **FIXED_HPARAMS,
        "seed": int(seed),
        "learning_rate": float(best["learning_rate"]),
        "num_transforms": int(best["num_transforms"]),
        "emb_dim": int(best["emb_dim"]),
        "gnn_hidden": int(best["gnn_hidden"]),
        "gnn_layers": int(best["gnn_layers"]),
        "gnn_k": int(best["gnn_k"]),
        "hidden_features": int(best.get("hidden_features", 64)),
    }


def build_embedding(variant, cfg, T_len, hps):
    common = dict(
        emb_dim=hps["emb_dim"],
        hidden=hps["gnn_hidden"],
        n_layers=hps["gnn_layers"],
        k=hps["gnn_k"],
        dropout=hps["dropout"],
    )
    if variant == "V0_knn_mean":
        return GNN(cfg["positions"], cfg["region_side"], T_len, **common)
    raise ValueError(f"Unknown trained variant: {variant}")


def compute_metrics(samples, theta, param_names):
    ok = np.isfinite(samples).all(axis=(1, 2))
    if not ok.any():
        raise ValueError("No finite posterior samples.")
    samples = samples[ok]
    theta = theta[ok]
    med = np.median(samples, axis=1)
    lo = np.quantile(samples, 0.025, axis=1)
    hi = np.quantile(samples, 0.975, axis=1)
    rows = []
    for j, name in enumerate(param_names):
        err = med[:, j] - theta[:, j]
        cover = (theta[:, j] >= lo[:, j]) & (theta[:, j] <= hi[:, j])
        rows.append({
            "parameter": name,
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "ci95_width": float(np.mean(hi[:, j] - lo[:, j])),
            "coverage_95": float(np.mean(cover)),
        })
    return rows


def check_hparams_match(scenario, seed, hps):
    meta_path = SEEDS_ROOT / f"seed_{seed}" / scenario / "final" / "gnn_run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Main GNN run missing: {meta_path}\nRun sim_04 first.")
    with open(meta_path) as f:
        main_hps = json.load(f)["hparams"]
    keys = ("learning_rate", "num_transforms", "emb_dim", "gnn_hidden", "gnn_layers", "gnn_k",
            "hidden_features", "seed")
    diff = [k for k in keys if main_hps.get(k) != hps.get(k)]
    if diff:
        raise ValueError(f"Hyperparameters differ from the main GNN run for {scenario} seed={seed}: {diff}")


def run_one(scenario, variant, seed, cfg, prior, data, saved):
    out_dir = OUT_ROOT / scenario / variant / f"seed_{seed}"
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        print(f"[skip] {scenario} {variant} seed={seed}")
        return

    th_tr, x_tr, th_te, x_te, T_len = data
    param_names = PARAM_NAMES[scenario]
    hps = gnn_hparams(saved, seed)
    check_hparams_match(scenario, seed, hps)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    emb = build_embedding(variant, cfg, T_len, hps).to(DEVICE)
    n_emb = count_params(emb)

    print(f"\n[{scenario}] {variant} seed={seed}  embedding params={n_emb}")
    posterior, info = train_npe(x_tr, th_tr, hps, prior, emb, DEVICE)
    print(f"  trained: epochs={info.get('epochs_run', '?')}  "
          f"best_val_lp={info.get('best_val_lp', float('nan')):.4f}  {info['train_sec']:.0f}s")

    net = density_estimator(posterior)
    n_total = count_params(net) if net is not None else None
    if net is not None:
        torch.save(net.state_dict(), out_dir / "density_estimator_state.pt")

    set_seed(seed)
    samples, infer_sec, ms_per_obs = run_inference(
        posterior,
        x_te,
        len(param_names),
        scenario,
        cfg["lut_betas"],
        cfg["lut_seed_rowsums"],
        cfg["log_dist"],
        cfg["seed_idx"],
        DEVICE,
    )
    np.savez_compressed(
        out_dir / "posterior_samples.npz",
        samples=samples,
        param_names=np.array(param_names, dtype=object),
    )

    metrics = compute_metrics(samples, th_te, param_names)
    meta = {
        "scenario": scenario,
        "variant": variant,
        "seed": int(seed),
        "param_names": list(param_names),
        "n_params_embedding": n_emb,
        "n_params_total": n_total,
        "training": {
            "epochs_run": info.get("epochs_run", 0),
            "best_epoch": info.get("best_epoch", -1),
            "best_val_lp": info.get("best_val_lp", float("nan")),
            "n_train": int(th_tr.shape[0]),
            "n_test": int(th_te.shape[0]),
        },
        "timing": {
            "train_sec": info["train_sec"],
            "infer_sec": infer_sec,
            "ms_per_obs": ms_per_obs,
        },
        "hparams": hps,
        "metrics": metrics,
    }
    tmp_path = out_dir / "meta.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_path.replace(meta_path)

    for m in metrics:
        print(f"  {m['parameter']:>6}: MAE={m['mae']:.4f}  width={m['ci95_width']:.4f}  "
              f"coverage={m['coverage_95']:.3f}")


def main_variant_records():
    records = []
    for meta_path in sorted(SEEDS_ROOT.glob("seed_*/*/final/gnn_run_meta.json")):
        final_dir = meta_path.parent
        with open(meta_path) as f:
            meta = json.load(f)
        scenario = meta["scenario"]
        if scenario not in SCENARIOS:
            continue
        samples = np.load(final_dir / "gnn_posterior_samples.npz", allow_pickle=True)["samples"]
        theta = np.load(final_dir / "theta_test.npy")
        lam = meta.get("learned_lambda_per_layer", [])
        records.append({
            "scenario": scenario,
            "variant": MAIN_VARIANT,
            "seed": int(meta["seed"]),
            "best_val_lp": meta["training"]["best_val_lp"],
            "n_params_embedding": meta["n_params_embedding"],
            "mean_lambda": float(np.mean(lam)) if lam else float("nan"),
            "metrics": compute_metrics(samples, theta, meta["param_names"]),
        })
    return records


def trained_variant_records():
    records = []
    for meta_path in sorted(OUT_ROOT.glob("*/*/seed_*/meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        records.append({
            "scenario": meta["scenario"],
            "variant": meta["variant"],
            "seed": int(meta["seed"]),
            "best_val_lp": meta["training"]["best_val_lp"],
            "n_params_embedding": meta["n_params_embedding"],
            "mean_lambda": float("nan"),
            "metrics": meta["metrics"],
        })
    return records


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    long_rows = []
    for rec in main_variant_records() + trained_variant_records():
        for m in rec["metrics"]:
            long_rows.append({
                "scenario": rec["scenario"],
                "variant": rec["variant"],
                "seed": rec["seed"],
                "parameter": m["parameter"],
                "mae": m["mae"],
                "bias": m["bias"],
                "ci95_width": m["ci95_width"],
                "coverage_95": m["coverage_95"],
                "best_val_lp": rec["best_val_lp"],
                "n_params_embedding": rec["n_params_embedding"],
                "mean_lambda": rec["mean_lambda"],
            })

    if not long_rows:
        print("No results found to aggregate.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_ROOT / "ablation_long.csv", long_rows)

    groups = {}
    for r in long_rows:
        groups.setdefault((r["scenario"], r["variant"], r["parameter"]), []).append(r)

    order_s = {s: i for i, s in enumerate(SCENARIOS)}
    order_v = {v: i for i, v in enumerate(VARIANTS)}
    summary_rows = []
    for key in sorted(groups, key=lambda k: (order_s.get(k[0], 99), order_v.get(k[1], 99))):
        rows = groups[key]
        out = {"scenario": key[0], "variant": key[1], "parameter": key[2], "n_seeds": len(rows)}
        for col in ("mae", "ci95_width", "coverage_95", "best_val_lp", "mean_lambda"):
            vals = np.array([r[col] for r in rows], dtype=float)
            fin = np.isfinite(vals)
            out[f"{col}_mean"] = float(vals[fin].mean()) if fin.any() else float("nan")
            out[f"{col}_sd"] = float(vals[fin].std(ddof=1)) if fin.sum() > 1 else float("nan")
        out["n_params_embedding"] = rows[0]["n_params_embedding"]
        summary_rows.append(out)
    write_csv(OUT_ROOT / "ablation_summary.csv", summary_rows)

    by_seed = {(r["scenario"], r["variant"], r["parameter"], r["seed"]): r for r in long_rows}
    diff_rows = []
    for scenario in SCENARIOS:
        for variant in TRAINED_VARIANTS:
            params = sorted({k[2] for k in by_seed if k[0] == scenario and k[1] == variant})
            for param in params:
                seeds = sorted(k[3] for k in by_seed
                               if k[0] == scenario and k[1] == variant and k[2] == param
                               and (scenario, MAIN_VARIANT, param, k[3]) in by_seed)
                if not seeds:
                    continue
                out = {"scenario": scenario, "comparison": f"{variant} - {MAIN_VARIANT}",
                       "parameter": param, "n_paired_seeds": len(seeds)}
                for col in ("mae", "ci95_width", "coverage_95"):
                    d = np.array([by_seed[(scenario, variant, param, s)][col]
                                  - by_seed[(scenario, MAIN_VARIANT, param, s)][col] for s in seeds])
                    out[f"{col}_diff_mean"] = float(d.mean())
                    out[f"{col}_diff_sd"] = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
                    if col == "mae":
                        base = np.array([by_seed[(scenario, MAIN_VARIANT, param, s)][col] for s in seeds])
                        out["mae_rel_change_mean"] = float(np.mean(d / base))
                diff_rows.append(out)
    if diff_rows:
        write_csv(OUT_ROOT / "ablation_paired_diff.csv", diff_rows)

    print(f"\nSaved results in {OUT_ROOT}")
    for r in summary_rows:
        print(f"{r['scenario']:>8} {r['variant']:>16} {r['parameter']:>6}  "
              f"MAE={r['mae_mean']:.4f}±{r['mae_sd']:.4f}  "
              f"width={r['ci95_width_mean']:.4f}±{r['ci95_width_sd']:.4f}  "
              f"cov={r['coverage_95_mean']:.3f}±{r['coverage_95_sd']:.3f}  (n={r['n_seeds']})")
    for r in diff_rows:
        print(f"{r['scenario']:>8} {r['comparison']:>34} {r['parameter']:>6}  "
              f"dMAE={r['mae_diff_mean']:+.4f}±{r['mae_diff_sd']:.4f} ({r['mae_rel_change_mean']:+.1%})  "
              f"dCov={r['coverage_95_diff_mean']:+.3f}  (n={r['n_paired_seeds']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    parser.add_argument("--variants", nargs="+", default=list(TRAINED_VARIANTS), choices=list(TRAINED_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    cfg = load_config()
    t_start = time.perf_counter()
    failures = []

    for scenario in scenarios:
        scenario_dir = STRICT_ROOT / scenario
        with open(scenario_dir / "best_hparams.json") as f:
            saved = json.load(f)
        prior = make_prior(scenario, cfg["lut_betas"], cfg["lut_seed_rowsums"],
                           cfg["log_dist"], cfg["seed_idx"])
        stored = [str(v) for v in np.load(scenario_dir / "final" / "test.npz", allow_pickle=True)["param_names"]]
        if stored != list(PARAM_NAMES[scenario]):
            raise ValueError(f"Parameter order mismatch for {scenario}: {stored}")
        data = load_gnn(scenario_dir / "final")
        print(f"\n{'=' * 60}\nscenario={scenario}  n_train={data[0].shape[0]}  "
              f"n_test={data[2].shape[0]}  device={DEVICE}")
        for seed in args.seeds:
            for variant in args.variants:
                try:
                    run_one(scenario, variant, seed, cfg, prior, data, saved)
                except Exception as exc:
                    failures.append({"scenario": scenario, "variant": variant,
                                     "seed": int(seed), "error": repr(exc)})
                    print(f"[failed] {scenario} {variant} seed={seed}: {exc!r}")

    print(f"\nTotal wall time: {(time.perf_counter() - t_start) / 60:.1f} min")
    if failures:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        tag = f"{args.scenario}_{'-'.join(str(s) for s in args.seeds)}"
        fail_path = OUT_ROOT / f"failed_runs_{tag}.json"
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"{len(failures)} run(s) failed; see {fail_path}")


if __name__ == "__main__":
    main()
