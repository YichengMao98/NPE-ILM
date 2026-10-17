import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.stats import norm, rankdata

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()
OUT_ROOT = ARC_ROOT / "mcmc_diagnostics"

SCENARIOS = ("full", "gamma", "partial")
RHAT_THRESHOLD = 1.01
ESS_MIN = 400
EXCLUDED_FROM_CRITERION = ("logpost",)


def autocov(x):
    n = x.shape[-1]
    m = next_fast_len(2 * n)
    xc = x - x.mean(axis=-1, keepdims=True)
    f = np.fft.rfft(xc, n=m, axis=-1)
    return np.fft.irfft(f * np.conjugate(f), n=m, axis=-1)[..., :n] / n


def ess_chains(x):
    n_chain, n_draw = x.shape
    if n_draw < 4 or not np.all(np.isfinite(x)):
        return float("nan")
    acov = autocov(x)
    chain_mean = x.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, ddof=1)
    if var_plus <= 0:
        return float("nan")
    rho = np.zeros(n_draw)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n_draw - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    total = n_chain * n_draw
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1: max_t + 2])
    tau = max(tau, 1.0 / np.log10(total))
    return float(total / tau)


def split_chains(x):
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def rank_normalize(x):
    r = rankdata(x, method="average").reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def rhat_basic(x):
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1))
    if within <= 0:
        return float("nan")
    between = n * np.var(x.mean(axis=1), ddof=1)
    return float(np.sqrt((between / within + n - 1) / n))


def rhat_rank(x):
    s = split_chains(x)
    bulk = rhat_basic(rank_normalize(s))
    folded = rhat_basic(rank_normalize(np.abs(s - np.median(s))))
    return float(np.nanmax([bulk, folded]))


def ess_bulk(x):
    return ess_chains(rank_normalize(split_chains(x)))


def ess_tail(x):
    s = split_chains(x)
    q05, q95 = np.quantile(s, [0.05, 0.95])
    lo = ess_chains((s <= q05).astype(float))
    hi = ess_chains((s <= q95).astype(float))
    return float(np.nanmin([lo, hi]))


def quantity_kind(name, param_names):
    if name in param_names:
        return "parameter"
    if name in ("loglik", "logpost"):
        return name
    return "latent"


def diagnose_scenario(scenario):
    mcmc_dir = STRICT_ROOT / scenario / "mcmc"
    with open(mcmc_dir / "mcmc_run_meta.json") as f:
        run_meta = json.load(f)
    n_obs = int(run_meta["n_obs"])
    param_names = list(run_meta["param_names"])
    obs_idx = np.load(mcmc_dir / "mcmc_obs_idx.npy").astype(int)

    rows = []
    for i in range(n_obs):
        with open(mcmc_dir / f"obs_{i:03d}_meta.json") as f:
            meta = json.load(f)
        if int(meta["obs_id"]) != int(obs_idx[i]):
            raise ValueError(f"{scenario} obs_{i:03d}: obs_id does not match mcmc_obs_idx.npy")
        for name, d in meta["diagnostics"].items():
            rows.append({
                "scenario": scenario,
                "local_mcmc_index": i,
                "obs_idx": int(obs_idx[i]),
                "quantity": name,
                "kind": quantity_kind(name, param_names),
                "rhat_rank": d["rhat_rank"],
                "ess_bulk": d["ess_bulk"],
                "ess_tail": d["ess_tail"],
                "cpu_sec": meta["timing"]["cpu_sec"],
                "wall_sec": meta["timing"]["wall_sec"],
            })
    by_q = pd.DataFrame(rows)
    by_q["ess_bulk_per_cpu_sec"] = by_q["ess_bulk"] / by_q["cpu_sec"]
    by_q["in_criterion"] = ~by_q["quantity"].isin(EXCLUDED_FROM_CRITERION)
    by_q["passes"] = (by_q["rhat_rank"] < RHAT_THRESHOLD) & (by_q["ess_bulk"] > ESS_MIN) & (by_q["ess_tail"] > ESS_MIN)

    crit = by_q[by_q["in_criterion"]]
    by_obs = crit.groupby(["scenario", "local_mcmc_index", "obs_idx"]).agg(
        max_rhat_rank=("rhat_rank", "max"),
        min_ess_bulk=("ess_bulk", "min"),
        min_ess_tail=("ess_tail", "min"),
        converged=("passes", "all"),
        cpu_sec=("cpu_sec", "first"),
        wall_sec=("wall_sec", "first"),
    ).reset_index()
    params_only = by_q[by_q["kind"] == "parameter"].groupby("local_mcmc_index")["passes"].all()
    by_obs["parameters_pass"] = by_obs["local_mcmc_index"].map(params_only).astype(bool)
    return by_q, by_obs


def summarize(by_q, by_obs):
    rows = []
    for (scenario, name), g in by_q.groupby(["scenario", "quantity"], sort=False):
        row = {"scenario": scenario, "quantity": name, "kind": g["kind"].iloc[0], "n_obs": int(len(g))}
        for c in ("ess_bulk", "ess_tail", "ess_bulk_per_cpu_sec"):
            v = g[c].to_numpy(float)
            row[f"{c}_median"] = float(np.nanmedian(v))
            row[f"{c}_q05"] = float(np.nanquantile(v, 0.05))
            row[f"{c}_min"] = float(np.nanmin(v))
        row["rhat_rank_median"] = float(g["rhat_rank"].median())
        row["rhat_rank_max"] = float(g["rhat_rank"].max())
        row["pct_pass"] = float(100 * g["passes"].mean())
        rows.append(row)
    per_q = pd.DataFrame(rows)
    per_s = by_obs.groupby("scenario", sort=False).agg(
        n_obs=("converged", "size"),
        n_converged=("converged", "sum"),
        n_parameters_pass=("parameters_pass", "sum"),
        median_wall_min=("wall_sec", lambda v: float(np.median(v)) / 60),
        median_cpu_min=("cpu_sec", lambda v: float(np.median(v)) / 60),
    ).reset_index()
    return per_q, per_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_q, all_obs = [], []
    for scenario in scenarios:
        by_q, by_obs = diagnose_scenario(scenario)
        by_q.to_csv(OUT_ROOT / f"diagnostics_by_obs_quantity_{scenario}.csv", index=False)
        by_obs.to_csv(OUT_ROOT / f"diagnostics_by_obs_{scenario}.csv", index=False)
        all_q.append(by_q)
        all_obs.append(by_obs)

    per_q, per_s = summarize(pd.concat(all_q, ignore_index=True), pd.concat(all_obs, ignore_index=True))
    tag = "" if args.scenario == "all" else f"_{args.scenario}"
    per_q.to_csv(OUT_ROOT / f"diagnostics_summary_by_quantity{tag}.csv", index=False)
    per_s.to_csv(OUT_ROOT / f"diagnostics_summary_by_scenario{tag}.csv", index=False)
    with open(OUT_ROOT / f"diagnostics_settings{tag}.json", "w") as f:
        json.dump({
            "source": "diagnostics written by sim_05_mcmc_da.py, computed on all post-burn-in draws of each chain",
            "rhat": "rank-normalized split R-hat, max of bulk and folded (Vehtari et al. 2021)",
            "ess_bulk": "rank-normalized split-chain ESS",
            "ess_tail": "min ESS of indicators for the 5% and 95% quantiles",
            "rhat_threshold": RHAT_THRESHOLD,
            "ess_min": ESS_MIN,
            "criterion": "all parameters, the log-likelihood and the latent summary pass all three checks",
            "excluded_from_criterion": list(EXCLUDED_FROM_CRITERION),
        }, f, indent=2)

    pd.set_option("display.width", 250)
    print("Per scenario")
    print(per_s.round(2).to_string(index=False))
    print("\nPer quantity")
    cols = ["scenario", "quantity", "ess_bulk_median", "ess_bulk_min", "ess_tail_median", "ess_tail_min", "rhat_rank_max", "pct_pass"]
    print(per_q[cols].round(3).to_string(index=False))
    print(f"\nSaved outputs in {OUT_ROOT}")


if __name__ == "__main__":
    main()
