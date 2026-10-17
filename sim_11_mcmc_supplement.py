import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sim_06_eval_cnnvsgnn import load_config

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = ARC_ROOT / "strict"
SEEDS_ROOT = ARC_ROOT / "strict_seeds"
DIAG_ROOT = ARC_ROOT / "mcmc_diagnostics"
OUT_ROOT = ARC_ROOT / "mcmc_supplement"

SCENARIOS = ("full", "gamma", "partial")
SCENARIO_LABELS = {"full": "Full", "gamma": r"Stoch.\ removal", "partial": "Partial"}
PARAM_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$", "rho": r"$\rho$"}
MOVE_LABELS = {
    "alpha_beta": r"$(\log\alpha,\beta)$",
    "gamma_centered": r"$\gamma$, centred",
    "gamma_noncentered": r"$\gamma$, non-centred",
    "dur_prior": "Duration, draw from duration model",
    "dur_local": "Duration, local shift",
    "add": "Hidden infection, addition",
    "remove": "Hidden infection, removal",
    "shift_uniform": "Hidden infection, uniform shift",
    "shift_local": "Hidden infection, local shift",
}
EXAMPLE_SEED = 1


def discover_seeds():
    return sorted(int(m.group(1)) for p in SEEDS_ROOT.glob("seed_*")
                  if (m := re.fullmatch(r"seed_(\d+)", p.name)) and p.is_dir())


def load_diag(scenario):
    return pd.read_csv(DIAG_ROOT / f"diagnostics_by_obs_{scenario}.csv").sort_values("local_mcmc_index").reset_index(drop=True)


def load_metas(scenario, n):
    metas = []
    for i in range(n):
        with open(STRICT_ROOT / scenario / "mcmc" / f"obs_{i:03d}_meta.json") as f:
            metas.append(json.load(f))
    return metas


def fmt_range(v, digits, comma=False):
    v = np.asarray(v, float)
    f = (lambda a: f"{a:,.0f}") if comma else (lambda a: f"{a:.{digits}f}")
    return f"{f(np.median(v))} [{f(v.min())}, {f(v.max())}]"


def write_tex(path, colspec, header, body, caption, label):
    lines = [r"\begin{table}[h!]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += body + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def diagnostics_table():
    rows, body = [], []
    for sc in SCENARIOS:
        d = load_diag(sc)
        row = {
            "scenario": sc, "n_epidemics": len(d), "n_converged": int(d["converged"].sum()),
            "max_rhat_median": d["max_rhat_rank"].median(), "max_rhat_min": d["max_rhat_rank"].min(),
            "max_rhat_max": d["max_rhat_rank"].max(),
            "min_ess_bulk_median": d["min_ess_bulk"].median(), "min_ess_bulk_min": d["min_ess_bulk"].min(),
            "min_ess_bulk_max": d["min_ess_bulk"].max(),
            "min_ess_tail_median": d["min_ess_tail"].median(), "min_ess_tail_min": d["min_ess_tail"].min(),
            "min_ess_tail_max": d["min_ess_tail"].max(),
            "cpu_min_mean": d["cpu_sec"].mean() / 60.0,
        }
        rows.append(row)
        body.append(" & ".join([
            SCENARIO_LABELS[sc], f"{row['n_converged']}/{row['n_epidemics']}",
            fmt_range(d["max_rhat_rank"], 3), fmt_range(d["min_ess_bulk"], 0, True),
            fmt_range(d["min_ess_tail"], 0, True),
        ]) + r" \\")
    write_tex(
        OUT_ROOT / "table_mcmc_diagnostics.tex", "lcccc",
        r"Scenario & Converged & Max.\ $\hat{R}$ & Min.\ bulk ESS & Min.\ tail ESS",
        body,
        r"MCMC convergence diagnostics over the 100 epidemics per scenario. For each epidemic, the maximum "
        r"$\hat{R}$ and the minimum bulk and tail ESS are taken over the parameters, the log-likelihood and the "
        r"latent summary. Entries are the median [minimum, maximum] over epidemics.",
        "tab:si_mcmc_diagnostics",
    )
    return pd.DataFrame(rows)


def acceptance_table():
    rows, body = [], []
    for sc in SCENARIOS:
        d = load_diag(sc)
        metas = load_metas(sc, len(d))
        per_obs = pd.DataFrame([pd.DataFrame(m["acceptance"]).mean(axis=0) for m in metas])
        for i, move in enumerate(per_obs.columns):
            v = per_obs[move].to_numpy(float)
            rows.append({"scenario": sc, "move": move, "median": np.median(v), "min": v.min(), "max": v.max()})
            body.append(" & ".join([SCENARIO_LABELS[sc] if i == 0 else "", MOVE_LABELS.get(move, move),
                                    fmt_range(v, 2)]) + r" \\")
        if sc != SCENARIOS[-1]:
            body.append(r"\midrule")
    write_tex(
        OUT_ROOT / "table_mcmc_acceptance.tex", "llc",
        r"Scenario & Update & Acceptance rate",
        body,
        r"Post-burn-in acceptance rates of the MCMC updates. For each epidemic the rate is averaged over the four "
        r"chains. Entries are the median [minimum, maximum] over the 100 epidemics.",
        "tab:si_mcmc_acceptance",
    )
    return pd.DataFrame(rows)


def mcmc_flat(scenario):
    raw = np.load(STRICT_ROOT / scenario / "mcmc" / "mcmc_samples.npz", allow_pickle=True)["samples"]
    return raw.reshape(raw.shape[0], -1, raw.shape[-1]).astype(np.float64)


def group_metrics(err, inside, width):
    return {"mae": float(np.mean(err)), "width": float(np.median(width)), "coverage": float(np.mean(inside))}


def partial_by_convergence():
    sc = "partial"
    d = load_diag(sc)
    obs_idx = d["obs_idx"].to_numpy()
    conv = d["converged"].to_numpy(bool)
    theta = np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_theta_test.npy").astype(np.float64)
    names = ("alpha", "beta", "rho")
    rows = []
    for model, tag in (("CNN-NPE", "cnn"), ("GNN-NPE", "gnn")):
        for seed in discover_seeds():
            lv = pd.read_csv(SEEDS_ROOT / f"seed_{seed}" / sc / "eval" / f"param_level_{tag}.csv")
            lv = lv[lv["obs_id"].isin(obs_idx)].copy()
            lv["converged"] = lv["obs_id"].map(dict(zip(obs_idx, conv)))
            for (p, c), g in lv.groupby(["parameter", "converged"]):
                rows.append({"model": model, "seed": seed, "parameter": p, "converged": bool(c),
                             **group_metrics(g["abs_error"], g["coverage_95"], g["ci95_width"])})
    s = mcmc_flat(sc)
    med = np.median(s, axis=1)
    lo, hi = np.quantile(s, 0.025, axis=1), np.quantile(s, 0.975, axis=1)
    for j, p in enumerate(names):
        for c in (True, False):
            m = conv == c
            rows.append({"model": "MCMC", "seed": 0, "parameter": p, "converged": c,
                         **group_metrics(np.abs(med[m, j] - theta[m, j]),
                                         (lo[m, j] <= theta[m, j]) & (theta[m, j] <= hi[m, j]),
                                         hi[m, j] - lo[m, j])})
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["model", "parameter", "converged"]).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        width_mean=("width", "mean"), width_sd=("width", "std"),
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    def cell(r, m):
        mean, sd = r[f"{m}_mean"], r[f"{m}_sd"]
        return f"{mean:.3f}" if r["model"] == "MCMC" or not np.isfinite(sd) else f"{mean:.3f} ({sd:.3f})"

    body = []
    for i, model in enumerate(("CNN-NPE", "GNN-NPE", "MCMC")):
        for k, p in enumerate(names):
            cells = [model if k == 0 else "", PARAM_TEX[p]]
            for c in (True, False):
                r = agg[(agg["model"] == model) & (agg["parameter"] == p) & (agg["converged"] == c)].iloc[0]
                cells += [cell(r, "mae"), cell(r, "width"), cell(r, "coverage")]
            body.append(" & ".join(cells) + r" \\")
        if i < 2:
            body.append(r"\midrule")
    n_c, n_n = int(conv.sum()), int((~conv).sum())
    header = (rf"& & \multicolumn{{3}}{{c}}{{Converged ($n={n_c}$)}} & \multicolumn{{3}}{{c}}{{Not converged ($n={n_n}$)}} \\ "
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-8} Method & Parameter & MAE & Width & Coverage & MAE & Width & Coverage")
    write_tex(
        OUT_ROOT / "table_partial_by_convergence.tex", "llcccccc", header, body,
        r"Posterior performance in Partial for epidemics with converged and non-converged MCMC runs. Width is the "
        r"median width of the 95\% credible interval. NPE entries are the mean (SD) over five training runs.",
        "tab:si_partial_convergence",
    )
    per_seed.to_csv(OUT_ROOT / "partial_by_convergence_per_seed.csv", index=False)
    agg.to_csv(OUT_ROOT / "partial_by_convergence.csv", index=False)

    pd.DataFrame({"obs_idx": obs_idx, "rho_true": theta[:, 2], "converged": conv,
                  "max_rhat_rank": d["max_rhat_rank"], "parameters_pass": d["parameters_pass"]}).to_csv(
        OUT_ROOT / "partial_rho_convergence.csv", index=False)
    return agg


def hidden_infections():
    sc = "partial"
    cfg = load_config()
    d = load_diag(sc)
    obs_idx = d["obs_idx"].to_numpy()
    x = np.load(STRICT_ROOT / sc / "final" / "test.npz", allow_pickle=True)["x"].astype(int)
    stats = pd.read_csv(STRICT_ROOT / sc / "final" / "test_stats.csv")
    lat = np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_latent_raw.npz", allow_pickle=True)["latent"].astype(np.float64)
    n_total = np.rint(stats["attack_rate"].to_numpy()[obs_idx] * cfg["N"]).astype(int)
    n_obs = (x[obs_idx] >= 0).sum(axis=1)
    true_hidden = n_total - n_obs
    flat = lat.reshape(lat.shape[0], -1)
    q = np.quantile(flat, [0.025, 0.5, 0.975], axis=1)
    df = pd.DataFrame({
        "obs_idx": obs_idx, "converged": d["converged"].to_numpy(bool),
        "rho_true": np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_theta_test.npy")[:, 2],
        "n_observed": n_obs, "true_hidden": true_hidden,
        "post_q025": q[0], "post_median": q[1], "post_q975": q[2],
    })
    df["inside_95"] = (df["true_hidden"] >= df["post_q025"]) & (df["true_hidden"] <= df["post_q975"])
    df.to_csv(OUT_ROOT / "partial_hidden_infections.csv", index=False)
    return df


def nonconverged_example():
    sc = "partial"
    d = load_diag(sc)
    non = d[~d["converged"]].sort_values("max_rhat_rank").reset_index(drop=True)
    row = non.iloc[len(non) // 2]
    k, o = int(row["local_mcmc_index"]), int(row["obs_idx"])
    raw = np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_samples_raw.npz", allow_pickle=True)["samples"][k]
    lat = np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_latent_raw.npz", allow_pickle=True)["latent"][k]
    theta = np.load(STRICT_ROOT / sc / "mcmc" / "mcmc_theta_test.npy")[k]
    final_dir = SEEDS_ROOT / f"seed_{EXAMPLE_SEED}" / sc / "final"
    arrays = {
        "obs_idx": np.int32(o), "local_mcmc_index": np.int32(k), "theta": theta.astype(np.float32),
        "param_names": np.array(["alpha", "beta", "rho"], dtype=object),
        "max_rhat_rank": np.float32(row["max_rhat_rank"]),
        "mcmc_chains": raw.astype(np.float32), "mcmc_latent_chains": lat.astype(np.float32),
    }
    for tag in ("cnn", "gnn"):
        arrays[f"{tag}_samples"] = np.load(final_dir / f"{tag}_posterior_samples.npz")["samples"][o].astype(np.float32)
    np.savez_compressed(OUT_ROOT / "partial_nonconverged_example.npz", **arrays)
    return o, float(row["max_rhat_rank"])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diag = diagnostics_table()
    diag.to_csv(OUT_ROOT / "mcmc_diagnostics_summary.csv", index=False)
    acc = acceptance_table()
    acc.to_csv(OUT_ROOT / "mcmc_acceptance_summary.csv", index=False)
    agg = partial_by_convergence()
    hid = hidden_infections()
    o, rh = nonconverged_example()

    print(diag.round(3).to_string(index=False))
    print(acc.round(3).to_string(index=False))
    print(agg.round(3).to_string(index=False))
    for c, g in hid.groupby("converged"):
        print(f"hidden infections converged={c}: n={len(g)}  coverage={g['inside_95'].mean():.3f}  "
              f"MAE={np.abs(g['post_median'] - g['true_hidden']).mean():.1f}")
    print(f"non-converged example: obs_idx={o}  max R-hat={rh:.3f}")
    print(f"Saved outputs in {OUT_ROOT}")


if __name__ == "__main__":
    main()
