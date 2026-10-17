import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
SEEDS_ROOT = ARC_ROOT / "strict_seeds"
OUT_ROOT = ARC_ROOT / "multiseed_summary"

SCENARIOS = ("full", "gamma", "partial")
SCENARIO_LABELS = {"full": "Full", "gamma": "Stoch. removal", "partial": "Partial"}
PARAM_ORDER = ("alpha", "beta", "gamma", "rho")
PARAM_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$", "rho": r"$\rho$"}
MODEL_ORDER = ("CNN-NPE", "GNN-NPE", "MCMC")
MODEL_TAGS = {"CNN-NPE": "cnn", "GNN-NPE": "gnn"}
LEVEL_FILES = {
    "full_test": {
        "CNN-NPE": "param_level_cnn.csv",
        "GNN-NPE": "param_level_gnn.csv",
    },
    "mcmc_subset": {
        "CNN-NPE": "param_level_cnn_mcmc_subset.csv",
        "GNN-NPE": "param_level_gnn_mcmc_subset.csv",
        "MCMC": "param_level_mcmc.csv",
    },
}
COV_COLS = ("coverage_50", "coverage_80", "coverage_90", "coverage_95")
METRICS = ("mae", "rmse", "bias", "rel_error", "ci95_width_median", *COV_COLS, "pearson_r", "spearman_r")
DIFF_METRICS = ("mae", "rmse", "ci95_width_median", "coverage_95", "pearson_r")
BOOT_METRICS = {
    "mae": ("abs_error", "mean"),
    "ci95_width_median": ("ci95_width", "median"),
    "coverage_95": ("coverage_95", "mean"),
}
STAT_FUNCS = {"mean": np.mean, "median": np.median}
N_BOOT = 2000
BOOT_SEED = 20260930
VAL_LP_DROP = 0.5


def discover_seeds():
    seeds = []
    for p in SEEDS_ROOT.glob("seed_*"):
        m = re.fullmatch(r"seed_(\d+)", p.name)
        if m and p.is_dir():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def sort_frame(df, cols):
    key = {
        "scenario": lambda s: s.map({v: i for i, v in enumerate(SCENARIOS)}),
        "parameter": lambda s: s.map({v: i for i, v in enumerate(PARAM_ORDER)}),
        "model": lambda s: s.map({v: i for i, v in enumerate(MODEL_ORDER)}),
    }
    return df.sort_values(cols, key=lambda s: key[s.name](s) if s.name in key else s).reset_index(drop=True)


def load_levels(seeds):
    frames = []
    missing = []
    for seed in seeds:
        for scenario in SCENARIOS:
            eval_dir = SEEDS_ROOT / f"seed_{seed}" / scenario / "eval"
            for scope, files in LEVEL_FILES.items():
                for model, fname in files.items():
                    path = eval_dir / fname
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    df = pd.read_csv(path)
                    df["seed"] = seed
                    df["scenario"] = scenario
                    df["scope"] = scope
                    df["model"] = model
                    frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No evaluation files found under {SEEDS_ROOT}. Run sim_06 and sim_07 first.")
    return pd.concat(frames, ignore_index=True), missing


def check_same_truth(levels):
    for (scenario, scope, parameter), g in levels.groupby(["scenario", "scope", "parameter"]):
        spread = g.groupby("obs_id")["true"].agg(lambda v: float(v.max() - v.min()))
        if (spread > 1e-6).any():
            raise ValueError(f"True parameter values differ across seeds or models: {scenario} {scope} {parameter}")


def rank_corr(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_block(g):
    true = g["true"].to_numpy(float)
    med = g["post_median"].to_numpy(float)
    row = {
        "n_obs": int(len(g)),
        "mae": float(g["abs_error"].mean()),
        "rmse": float(np.sqrt(g["sq_error"].mean())),
        "bias": float(g["bias"].mean()),
        "rel_error": float(g["rel_error"].mean()),
        "ci95_width_median": float(g["ci95_width"].median()),
    }
    for c in COV_COLS:
        row[c] = float(g[c].mean())
    row["pearson_r"] = float(np.corrcoef(true, med)[0, 1])
    row["spearman_r"] = rank_corr(true, med)
    return row


def per_seed_metrics(levels):
    rows = []
    for (scenario, scope, model, seed, parameter), g in levels.groupby(
        ["scenario", "scope", "model", "seed", "parameter"]
    ):
        rows.append({
            "scenario": scenario,
            "scope": scope,
            "model": model,
            "seed": int(seed),
            "parameter": parameter,
            **summarize_block(g),
        })
    return sort_frame(pd.DataFrame(rows), ["scope", "scenario", "parameter", "model", "seed"])


def check_mcmc_identical(per_seed):
    mc = per_seed[per_seed["model"] == "MCMC"]
    for key, g in mc.groupby(["scenario", "parameter"]):
        vals = g[list(METRICS)].to_numpy(float)
        if not np.allclose(vals, vals[0], equal_nan=True):
            raise ValueError(f"MCMC metrics differ across seeds for {key}; MCMC should be seed independent.")


def summary_mean_sd(per_seed):
    rows = []
    for (scenario, scope, model, parameter), g in per_seed.groupby(["scenario", "scope", "model", "parameter"]):
        if model == "MCMC":
            g = g.iloc[:1]
        row = {
            "scenario": scenario,
            "scope": scope,
            "model": model,
            "parameter": parameter,
            "n_seeds": int(len(g)),
            "n_obs": int(g["n_obs"].iloc[0]),
        }
        for m in METRICS:
            v = g[m].to_numpy(float)
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            row[f"{m}_min"] = float(v.min())
            row[f"{m}_max"] = float(v.max())
        rows.append(row)
    return sort_frame(pd.DataFrame(rows), ["scope", "scenario", "parameter", "model"])


def paired_differences(per_seed):
    cnn = per_seed[per_seed["model"] == "CNN-NPE"]
    gnn = per_seed[per_seed["model"] == "GNN-NPE"]
    keys = ["scenario", "scope", "seed", "parameter"]
    merged = gnn.merge(cnn, on=keys, suffixes=("_gnn", "_cnn"))
    rows = []
    for (scenario, scope, parameter), g in merged.groupby(["scenario", "scope", "parameter"]):
        row = {
            "scenario": scenario,
            "scope": scope,
            "parameter": parameter,
            "comparison": "GNN-NPE minus CNN-NPE",
            "n_paired_seeds": int(len(g)),
        }
        for m in DIFF_METRICS:
            d = (g[f"{m}_gnn"] - g[f"{m}_cnn"]).to_numpy(float)
            row[f"{m}_diff_mean"] = float(d.mean())
            row[f"{m}_diff_sd"] = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        rel = (g["mae_gnn"] - g["mae_cnn"]) / g["mae_cnn"]
        row["mae_rel_change_mean"] = float(rel.mean())
        row["n_seeds_gnn_lower_mae"] = int((g["mae_gnn"] < g["mae_cnn"]).sum())
        rows.append(row)
    return sort_frame(pd.DataFrame(rows), ["scope", "scenario", "parameter"])


def seed_averaged(levels, scenario, scope, model, parameter, column):
    g = levels[
        (levels["scenario"] == scenario)
        & (levels["scope"] == scope)
        & (levels["model"] == model)
        & (levels["parameter"] == parameter)
    ]
    if g.empty:
        return None
    if model == "MCMC":
        g = g[g["seed"] == g["seed"].min()]
    wide = g.pivot_table(index="obs_id", columns="seed", values=column, aggfunc="first")
    wide = wide.dropna(axis=0, how="any")
    return wide.mean(axis=1).sort_index()


def bootstrap_ci(levels, n_boot):
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for scope in LEVEL_FILES:
        for scenario in SCENARIOS:
            sub = levels[(levels["scope"] == scope) & (levels["scenario"] == scenario)]
            if sub.empty:
                continue
            params = sorted(sub["parameter"].unique(), key=PARAM_ORDER.index)
            models = [m for m in MODEL_ORDER if m in set(sub["model"])]
            for parameter in params:
                for metric, (column, stat) in BOOT_METRICS.items():
                    f = STAT_FUNCS[stat]
                    series = {m: seed_averaged(levels, scenario, scope, m, parameter, column) for m in models}
                    series = {m: s for m, s in series.items() if s is not None}
                    if not series:
                        continue
                    common = sorted(set.intersection(*[set(s.index) for s in series.values()]))
                    vals = {m: s.loc[common].to_numpy(float) for m, s in series.items()}
                    n = len(common)
                    idx = rng.integers(0, n, size=(n_boot, n))
                    boots = {m: f(v[idx], axis=1) for m, v in vals.items()}
                    for m, v in vals.items():
                        lo, hi = np.quantile(boots[m], [0.025, 0.975])
                        rows.append({
                            "scenario": scenario, "scope": scope, "parameter": parameter,
                            "model": m, "metric": metric, "n_obs": n,
                            "estimate": float(f(v)), "ci_lo": float(lo), "ci_hi": float(hi),
                        })
                    if "CNN-NPE" in vals and "GNN-NPE" in vals:
                        d = boots["GNN-NPE"] - boots["CNN-NPE"]
                        lo, hi = np.quantile(d, [0.025, 0.975])
                        rows.append({
                            "scenario": scenario, "scope": scope, "parameter": parameter,
                            "model": "GNN-NPE minus CNN-NPE", "metric": metric, "n_obs": n,
                            "estimate": float(f(vals["GNN-NPE"]) - f(vals["CNN-NPE"])),
                            "ci_lo": float(lo), "ci_hi": float(hi),
                        })
    return pd.DataFrame(rows)


def training_meta(seeds):
    rows = []
    for seed in seeds:
        for scenario in SCENARIOS:
            final_dir = SEEDS_ROOT / f"seed_{seed}" / scenario / "final"
            for model, tag in MODEL_TAGS.items():
                path = final_dir / f"{tag}_run_meta.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    meta = json.load(f)
                tr = meta.get("training", {})
                tm = meta.get("timing", {})
                lam = meta.get("learned_lambda_per_layer", []) or []
                rows.append({
                    "scenario": scenario,
                    "model": model,
                    "seed": int(seed),
                    "n_params_embedding": meta.get("n_params_embedding"),
                    "n_params_total": meta.get("n_params_total"),
                    "best_val_lp": tr.get("best_val_lp", float("nan")),
                    "epochs_run": tr.get("epochs_run"),
                    "best_epoch": tr.get("best_epoch"),
                    "train_min": float(tm.get("train_sec", float("nan"))) / 60.0,
                    "infer_sec": tm.get("infer_sec", float("nan")),
                    "ms_per_obs": tm.get("ms_per_obs", float("nan")),
                    "mean_lambda": float(np.mean(lam)) if lam else float("nan"),
                    "lambda_per_layer": " ".join(f"{v:.3f}" for v in lam),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["val_lp_gap_to_median"] = df["best_val_lp"] - df.groupby(["scenario", "model"])["best_val_lp"].transform("median")
    df["val_lp_flag"] = df["val_lp_gap_to_median"] < -VAL_LP_DROP
    return sort_frame(df, ["scenario", "model", "seed"])


def training_summary(train):
    rows = []
    cols = ("n_params_embedding", "n_params_total", "best_val_lp", "epochs_run", "train_min", "ms_per_obs", "mean_lambda")
    for (scenario, model), g in train.groupby(["scenario", "model"]):
        row = {"scenario": scenario, "model": model, "n_seeds": int(len(g)), "n_flagged": int(g["val_lp_flag"].sum())}
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            row[f"{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[f"{c}_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
        rows.append(row)
    return sort_frame(pd.DataFrame(rows), ["scenario", "model"])


def fmt(mean, sd, digits=3):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def lookup(summary, scenario, scope, model, parameter):
    r = summary[
        (summary["scenario"] == scenario)
        & (summary["scope"] == scope)
        & (summary["model"] == model)
        & (summary["parameter"] == parameter)
    ]
    return None if r.empty else r.iloc[0]


def write_table(path, header, colspec, body_rows, caption, label):
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    lines += body_rows
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def main_table(summary, path):
    body = []
    for k, scenario in enumerate(SCENARIOS):
        params = [p for p in PARAM_ORDER if lookup(summary, scenario, "full_test", "GNN-NPE", p) is not None]
        for i, p in enumerate(params):
            cells = [SCENARIO_LABELS[scenario] if i == 0 else "", PARAM_TEX[p]]
            for m in ("mae", "ci95_width_median", "coverage_95"):
                for model in ("CNN-NPE", "GNN-NPE"):
                    r = lookup(summary, scenario, "full_test", model, p)
                    cells.append(fmt(r[f"{m}_mean"], r[f"{m}_sd"]) if r is not None else "--")
            body.append(" & ".join(cells) + r" \\")
        if k < len(SCENARIOS) - 1 and params:
            body.append(r"\midrule")
    header = (
        r"Scenario & Param. & \multicolumn{2}{c}{MAE} & \multicolumn{2}{c}{95\% CI width} "
        r"& \multicolumn{2}{c}{95\% coverage} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8} "
        r"& & CNN & GNN & CNN & GNN & CNN & GNN"
    )
    write_table(
        path, header, "llcccccc", body,
        "Parameter recovery on the full test set. CI width is the median width across test epidemics. "
        "Entries are mean (SD) across independently trained networks.",
        "tab:multiseed_main",
    )


def mcmc_table(summary, path):
    body = []
    for k, scenario in enumerate(SCENARIOS):
        params = [p for p in PARAM_ORDER if lookup(summary, scenario, "mcmc_subset", "MCMC", p) is not None]
        first = True
        for model in ("CNN-NPE", "GNN-NPE", "MCMC"):
            for i, p in enumerate(params):
                r = lookup(summary, scenario, "mcmc_subset", model, p)
                cells = [
                    SCENARIO_LABELS[scenario] if first else "",
                    model if i == 0 else "",
                    PARAM_TEX[p],
                ]
                first = False
                for m in ("mae", "spearman_r", "ci95_width_median", "coverage_95"):
                    cells.append(fmt(r[f"{m}_mean"], r[f"{m}_sd"]) if r is not None else "--")
                body.append(" & ".join(cells) + r" \\")
        if k < len(SCENARIOS) - 1 and params:
            body.append(r"\midrule")
    header = r"Scenario & Method & Parameter & MAE & $r_S$ & Width & Coverage"
    write_table(
        path, header, "lllcccc", body,
        "Comparison on the converged MCMC subset. $r_S$ is the Spearman correlation between true values and "
        "posterior medians. Width is the median width of the 95\\% credible interval across epidemics. "
        "NPE entries are mean (SD) across independently trained networks; MCMC does not depend on the network seed.",
        "tab:multiseed_mcmc",
    )


def training_table(tsum, path):
    body = []
    for scenario in SCENARIOS:
        for i, model in enumerate(("CNN-NPE", "GNN-NPE")):
            r = tsum[(tsum["scenario"] == scenario) & (tsum["model"] == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(" & ".join([
                SCENARIO_LABELS[scenario] if i == 0 else "",
                model,
                f"{int(round(r['n_params_embedding_mean'])):,}",
                f"{int(round(r['n_params_total_mean'])):,}" if np.isfinite(r["n_params_total_mean"]) else "--",
                fmt(r["best_val_lp_mean"], r["best_val_lp_sd"], 2),
                fmt(r["train_min_mean"], r["train_min_sd"], 1),
                fmt(r["ms_per_obs_mean"], r["ms_per_obs_sd"], 1),
            ]) + r" \\")
    header = r"Scenario & Model & Embedding params & Total params & Best val. log-prob & Training (min) & Inference (ms/obs)"
    write_table(
        path, header, "llrrccc", body,
        "Network size, training and inference cost. Entries are mean (SD) across independently trained networks.",
        "tab:multiseed_training",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else discover_seeds()
    if not seeds:
        raise FileNotFoundError(f"No seed directories found in {SEEDS_ROOT}")
    print(f"seeds: {seeds}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    levels, missing = load_levels(seeds)
    if missing:
        print(f"WARNING: {len(missing)} evaluation file(s) missing:")
        for m in missing:
            print(f"  {m}")
    check_same_truth(levels)

    per_seed = per_seed_metrics(levels)
    check_mcmc_identical(per_seed)
    summary = summary_mean_sd(per_seed)
    diffs = paired_differences(per_seed)
    boot = bootstrap_ci(levels, args.n_boot)
    train = training_meta(seeds)
    tsum = training_summary(train) if not train.empty else pd.DataFrame()

    per_seed.to_csv(OUT_ROOT / "per_seed_metrics.csv", index=False)
    summary.to_csv(OUT_ROOT / "summary_mean_sd.csv", index=False)
    diffs.to_csv(OUT_ROOT / "paired_diff_gnn_minus_cnn.csv", index=False)
    boot.to_csv(OUT_ROOT / "bootstrap_ci.csv", index=False)
    if not train.empty:
        train.to_csv(OUT_ROOT / "training_per_seed.csv", index=False)
        tsum.to_csv(OUT_ROOT / "training_summary.csv", index=False)
        training_table(tsum, OUT_ROOT / "table_training.tex")
    main_table(summary, OUT_ROOT / "table_main.tex")
    if (summary["scope"] == "mcmc_subset").any():
        mcmc_table(summary, OUT_ROOT / "table_mcmc_subset.tex")

    print(f"\nSaved outputs in {OUT_ROOT}")

    print("\nFull test set: mean (SD) across seeds")
    view = summary[summary["scope"] == "full_test"]
    for _, r in view.iterrows():
        print(f"  {r['scenario']:>8} {r['parameter']:>6} {r['model']:>8}  "
              f"MAE={fmt(r['mae_mean'], r['mae_sd'])}  width={fmt(r['ci95_width_median_mean'], r['ci95_width_median_sd'])}  "
              f"cov95={fmt(r['coverage_95_mean'], r['coverage_95_sd'])}  r={fmt(r['pearson_r_mean'], r['pearson_r_sd'])}  "
              f"(n={r['n_seeds']})")

    print("\nPaired GNN minus CNN (full test set)")
    for _, r in diffs[diffs["scope"] == "full_test"].iterrows():
        print(f"  {r['scenario']:>8} {r['parameter']:>6}  dMAE={fmt(r['mae_diff_mean'], r['mae_diff_sd'])}  "
              f"({r['mae_rel_change_mean']:+.1%})  GNN lower in {r['n_seeds_gnn_lower_mae']}/{r['n_paired_seeds']} seeds")

    if not train.empty and train["val_lp_flag"].any():
        print(f"\nRuns with best validation log-prob more than {VAL_LP_DROP} below the median of the other seeds:")
        for _, r in train[train["val_lp_flag"]].iterrows():
            print(f"  {r['scenario']} {r['model']} seed={r['seed']}  best_val_lp={r['best_val_lp']:.3f}")


if __name__ == "__main__":
    main()
