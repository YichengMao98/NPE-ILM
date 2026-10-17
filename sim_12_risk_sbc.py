import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from sim_06_eval_cnnvsgnn import IP_FIXED, load_config

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = ARC_ROOT / "strict"
SEEDS_ROOT = ARC_ROOT / "strict_seeds"
PPC_ROOT = ARC_ROOT / "individual_ppc"
OUT_ROOT = ARC_ROOT / "spatial_risk"
SBC_ROOT = ARC_ROOT / "sbc"

SCENARIO = "full"
MAP_SEED = 1
MAP_QUANTILE = 0.5
MAP_MIN_INFECTIOUS = 10
N_MAP_DRAWS = 1000
DRAW_SEED = 20261011
EPS = 1e-12
METHOD_KEYS = {"CNN-NPE": "cnn", "GNN-NPE": "gnn", "MCMC": "mcmc"}
SBC_SCENARIOS = ("full", "gamma", "partial")
SBC_BINS = 20
SBC_LEVEL = 0.99


def state_at(x, t):
    inf = np.where((x >= 0) & (x <= t) & (t - x < IP_FIXED))[0]
    sus = np.where((x < 0) | (x > t))[0]
    nxt = sus[x[sus] == t + 1]
    return inf, sus, nxt


def infection_prob(theta, log_dist, inf, sus):
    lk = log_dist[np.ix_(inf, sus)]
    pressure = np.exp(-theta[:, 1][:, None, None] * lk[None, :, :]).sum(axis=1)
    return -np.expm1(-theta[:, 0][:, None] * pressure)


def discover_seeds():
    return sorted(int(m.group(1)) for p in SEEDS_ROOT.glob("seed_*")
                  if (m := re.fullmatch(r"seed_(\d+)", p.name)) and p.is_dir())


def run_sbc():
    SBC_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = discover_seeds()
    edges = np.linspace(0, 1, SBC_BINS + 1)
    rows = []
    for sc in SBC_SCENARIOS:
        te = np.load(STRICT_ROOT / sc / "final" / "test.npz", allow_pickle=True)
        theta = te["theta"].astype(np.float64)
        names = [str(p) for p in te["param_names"]]
        n = len(theta)
        lo, hi = binom.ppf([(1 - SBC_LEVEL) / 2, (1 + SBC_LEVEL) / 2], n, 1 / SBC_BINS)
        out = {"edges": edges, "band_lower": np.float64(lo), "band_upper": np.float64(hi), "n_epidemics": np.int32(n),
               "param_names": np.array(names, dtype=object), "seeds": np.array(seeds)}
        for tag in ("cnn", "gnn"):
            counts_all = []
            for seed in seeds:
                s = np.load(SEEDS_ROOT / f"seed_{seed}" / sc / "final" / f"{tag}_posterior_samples.npz")["samples"]
                if s.shape[0] != n:
                    raise ValueError(f"Sample count mismatch for {sc} {tag} seed {seed}")
                ranks = (s < theta[:, None, :]).sum(axis=1)
                u = (ranks + 0.5) / (s.shape[1] + 1)
                counts = np.stack([np.histogram(u[:, j], bins=edges)[0] for j in range(len(names))])
                counts_all.append(counts)
                out[f"{tag}_seed{seed}_ranks"] = ranks.astype(np.int32)
                out["n_post"] = np.int32(s.shape[1])
                for j, p in enumerate(names):
                    rows.append({"scenario": sc, "model": tag.upper() + "-NPE", "seed": seed, "parameter": p,
                                 "n_bins_outside": int(((counts[j] < lo) | (counts[j] > hi)).sum())})
            out[f"{tag}_counts_mean"] = np.mean(counts_all, axis=0)
        np.savez_compressed(SBC_ROOT / f"sbc_{sc}.npz", **out)
    df = pd.DataFrame(rows)
    df.to_csv(SBC_ROOT / "sbc_summary_per_seed.csv", index=False)
    summ = df.groupby(["scenario", "model", "parameter"], sort=False).agg(
        n_seeds=("seed", "nunique"), n_bins_outside_mean=("n_bins_outside", "mean"),
        n_bins_outside_max=("n_bins_outside", "max")).reset_index()
    summ.to_csv(SBC_ROOT / "sbc_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))
    print(f"Saved SBC outputs in {SBC_ROOT}")


def run_risk_map():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    positions = np.load(ARC_ROOT / "configs" / "ilm_config.npz", allow_pickle=True)["positions"]
    x_all = np.load(STRICT_ROOT / SCENARIO / "final" / "test.npz", allow_pickle=True)["x"].astype(int)

    sel = pd.read_csv(PPC_ROOT / SCENARIO / "selection.csv")
    row = sel.iloc[(sel["quantile"] - MAP_QUANTILE).abs().argmin()]
    o, k = int(row["obs_idx"]), int(row["local_mcmc_index"])
    mcmc_dir = STRICT_ROOT / SCENARIO / "mcmc"
    if int(np.load(mcmc_dir / "mcmc_obs_idx.npy")[k]) != o:
        raise ValueError("Selected epidemic does not match the MCMC subset")
    raw = np.load(mcmc_dir / "mcmc_samples.npz", allow_pickle=True)["samples"][k].astype(np.float64)
    final_dir = SEEDS_ROOT / f"seed_{MAP_SEED}" / SCENARIO / "final"
    sets = {
        "CNN-NPE": np.load(final_dir / "cnn_posterior_samples.npz")["samples"][o].astype(np.float64),
        "GNN-NPE": np.load(final_dir / "gnn_posterior_samples.npz")["samples"][o].astype(np.float64),
        "MCMC": raw.reshape(-1, raw.shape[-1]),
    }

    x = x_all[o]
    n_inf = np.array([len(state_at(x, t)[0]) for t in range(int(x.max()))])
    t = int(np.argmax(n_inf >= MAP_MIN_INFECTIOUS))
    inf, sus, nxt = state_at(x, t)
    out = {"obs_idx": np.int32(o), "t": np.int32(t), "positions": positions.astype(np.float32),
           "infected_before": np.where((x >= 0) & (x <= t))[0], "infectious": inf, "susceptible": sus,
           "exposed_next": nxt}
    rng = np.random.default_rng(DRAW_SEED)
    rows = []
    for m, s in sets.items():
        th = s[rng.choice(len(s), size=min(N_MAP_DRAWS, len(s)), replace=False)]
        prob = infection_prob(th, cfg["log_dist"], inf, sus)
        lo, med, hi = np.quantile(prob, [0.025, 0.5, 0.975], axis=0)
        key = METHOD_KEYS[m]
        out[f"{key}_q025"], out[f"{key}_median"], out[f"{key}_q975"] = lo, med, hi
        ratio = hi / np.maximum(lo, EPS)
        rows.append({"method": m, "obs_idx": o, "t": t, "n_infectious": len(inf), "n_susceptible": len(sus),
                     "n_infected_next": len(nxt), "expected_infected_next": float(prob.sum(axis=1).mean()),
                     "ratio_q10": float(np.quantile(ratio, 0.1)), "ratio_median": float(np.median(ratio)),
                     "ratio_q90": float(np.quantile(ratio, 0.9))})
    np.savez_compressed(OUT_ROOT / "risk_map.npz", **out)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "risk_map_summary.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"Saved risk map outputs in {OUT_ROOT}")


def main():
    run_risk_map()
    run_sbc()


if __name__ == "__main__":
    main()
