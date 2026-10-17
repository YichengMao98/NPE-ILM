import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from fmd_02_make_datasets import DATA_ROOT, OUT_ROOT, PARAM_NAMES, draw_prior, lambda0, load_config, mean_ip, simulate

EVAL_ROOT = OUT_ROOT / "eval"
NPE_ROOT = OUT_ROOT / "npe"
MCMC_ROOT = OUT_ROOT / "mcmc"
SI_ROOT = OUT_ROOT / "si_local"
PAPER_ROOT = OUT_ROOT / "paper"
TAB_ROOT = PAPER_ROOT / "tables"
DATA_PATH = OUT_ROOT.parent / "fmd_clean.txt"

SEEDS = (1, 2, 3, 4, 5)
MODELS = ("cnn", "gnn")
LABEL = {"prior": "Prior", "mcmc": "MCMC", "cnn": "CNN-NPE", "gnn": "GNN-NPE"}
N_PRIOR = 20000
PRIOR_SEED = 97
RISK_DAY = 10
N_PRIOR_PRED = 1000
RISK_CHUNK = 500
SPATIAL_SEED = 2024
PPC_SETS = {"cnn": [f"cnn_seed{s}" for s in SEEDS], "gnn": [f"gnn_seed{s}" for s in SEEDS],
            "mcmc": ["mcmc_seed0"], "prior": ["prior_seed0"]}
N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))
QUANT_TEX = {"alpha0": r"$\alpha_0$", "alpha1": r"$\alpha_1$", "beta": r"$\beta$", "r0": r"$R_0$", "r": r"$r$",
             "loglik": r"$\log p(y\mid\vartheta)$", "logpost": r"$\log p(\vartheta\mid y)$"}


def derived(theta, cfg):
    theta = np.asarray(theta, dtype=np.float64)
    a_high = theta[:, 0] + theta[:, 1]
    r = theta[:, 0] / a_high
    lb = cfg["lut_betas"]
    b = theta[:, 2]
    inside = (b >= lb[0]) & (b <= lb[-1])
    lam = r * np.interp(b, lb, cfg["lut_all"]) + (1.0 - r) * np.interp(b, lb, cfg["lut_z1"])
    for i in np.where(~inside)[0]:
        lam[i] = lambda0(b[i], r[i], cfg)
    return a_high * mean_ip(cfg) * lam, r


def load_sets(cfg):
    sets = {"mcmc": np.load(MCMC_ROOT / "real.npz", allow_pickle=True)["samples_compare"].astype(np.float64)}
    for m in MODELS:
        for s in SEEDS:
            sets[f"{m}_seed{s}"] = np.load(NPE_ROOT / f"seed_{s}" / f"{m}_real_posterior.npz")["samples"].astype(np.float64)
        sets[m] = np.concatenate([sets[f"{m}_seed{s}"] for s in SEEDS])
    rng = np.random.default_rng(PRIOR_SEED)
    sets["prior"] = np.stack([draw_prior(rng, cfg)[0] for _ in range(N_PRIOR)])
    out = {}
    for k, th in sets.items():
        r0, r = derived(th, cfg)
        out[k] = np.column_stack([th, r0, r])
    return out


def sig(v, n=3):
    if v == 0 or not np.isfinite(v):
        return f"{v:.{n}g}"
    d = max(0, n - 1 - int(np.floor(np.log10(abs(v)))))
    return f"{v:.{d}f}"


def fmt_q(x, col):
    lo, med, hi = np.quantile(x, [0.025, 0.5, 0.975])
    if col in ("alpha0", "alpha1"):
        f = lambda v: sig(v, 2)
    else:
        f = lambda v: f"{v:.2f}"
    return f"{f(med)} [{f(lo)}, {f(hi)}]"


def num(v, d=2):
    return f"${v:.{d}f}$" if v < 0 else f"{v:.{d}f}"


def fmt_range(v, d=3):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return f"{lo:.{d}f}" if np.isclose(lo, hi) else f"[{lo:.{d}f}, {hi:.{d}f}]"


def write(name, body):
    TAB_ROOT.mkdir(parents=True, exist_ok=True)
    (TAB_ROOT / name).write_text(body)
    print(f"\n% ===== {name}\n{body}")


def table(caption, label, colspec, header, rows, env="table"):
    lines = [rf"\begin{{{env}}}[h!]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""]
    return "\n".join(lines)


COLS = ("alpha0", "alpha1", "beta", "r0", "r")


def tab_posterior(sets, cfg):
    rows = [[LABEL[k]] + [fmt_q(sets[k][:, j], c) for j, c in enumerate(COLS)] for k in ("mcmc", "cnn", "gnn")]
    cap = (r"Posterior medians and 95\% credible intervals for the observed outbreak. NPE results pool 3,000 draws "
           r"from each of five trained networks. $R_0$ follows \eqref{eq:fmd_r0}.")
    header = "Method & " + " & ".join(QUANT_TEX[c] for c in COLS)
    write("tab_fmd_posterior.tex", table(cap, "tab:fmd_posterior", "l" + "c" * len(COLS), header, rows))


_SP = None


def _init_spatial():
    global _SP
    cfg = load_config()
    cfg["log_kernel"] = cfg["log_kernel"].astype(np.float64)
    pos = cfg["positions"].astype(np.float64)
    _SP = {"cfg": cfg, "dist": np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))}


def infection_history(exp, rem, cfg):
    n = len(exp)
    L = int(cfg["latent"])
    onset = np.full(n, 10**6, dtype=np.int64)
    end = np.full(n, 10**6, dtype=np.int64)
    for key in ("active", "removed"):
        idx = cfg[f"{key}_idx"]
        onset[idx] = cfg[f"{key}_exp"] + L
        end[idx] = cfg[f"{key}_rem"]
    idx = cfg["exposed_idx"]
    onset[idx] = cfg["exposed_exp"] + L
    end[idx] = np.where(rem[idx] >= 0, rem[idx], 10**6)
    w = np.where(exp >= 1)[0]
    onset[w] = exp[w] + L
    has = w[rem[w] >= 0]
    end[has] = rem[has]
    return onset, end


def nearest_infectious_distance(exp, rem, cfg, dist):
    onset, end = infection_history(exp, rem, cfg)
    d = []
    for j in np.where(exp >= 1)[0]:
        t = exp[j] - 1
        inf = np.where((onset <= t) & (end > t))[0]
        if inf.size:
            d.append(dist[j, inf].min())
    return float(np.mean(d)) if d else np.nan


def _spatial_job(args):
    theta, seed = args
    rng = np.random.default_rng(seed)
    cfg, dist = _SP["cfg"], _SP["dist"]
    out = np.empty(len(theta))
    for i, th in enumerate(theta):
        e, r = simulate(th, cfg, rng, cfg["log_kernel"])
        out[i] = nearest_infectious_distance(e.astype(np.int64), r.astype(np.int64), cfg, dist)
    return out


def spatial_ppc(cfg, n_procs):
    jobs, owner = [], []
    for m, keys in PPC_SETS.items():
        for k in keys:
            th = np.load(EVAL_ROOT / f"ppc_{k}.npz")["theta"].astype(np.float64)
            for part in np.array_split(th, 4):
                jobs.append(part)
                owner.append(m)
    seeds = np.random.SeedSequence(SPATIAL_SEED).spawn(len(jobs))
    with Pool(processes=n_procs, initializer=_init_spatial) as pool:
        res = pool.map(_spatial_job, list(zip(jobs, seeds)))
    out = {m: np.concatenate([r for r, o in zip(res, owner) if o == m]) for m in PPC_SETS}
    pos = cfg["positions"].astype(np.float64)
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    out["observed"] = np.float64(nearest_infectious_distance(cfg["exp_obs"].astype(np.int64),
                                                             cfg["rem_obs"].astype(np.int64), cfg, dist))
    np.savez_compressed(PAPER_ROOT / "ppc_spatial.npz", **out)
    return out


def tab_ip():
    d = np.loadtxt(DATA_PATH)
    cull, onset, rem = d[:, 3], d[:, 5], d[:, 6]
    inf = onset > 0
    bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 1000)]
    rows = []
    for lo, hi in bins:
        m = inf & (onset > lo) & (onset <= hi)
        ip = (rem - onset)[m].astype(int)
        dl = (cull - onset)[m]
        cnt = np.bincount(ip, minlength=5)[1:5]
        lab = rf"$\leq {hi}$" if lo == 0 else (rf"$> {lo}$" if hi >= 1000 else rf"$({lo}, {hi}]$")
        rows.append([lab, str(int(m.sum()))] + [str(int(c)) for c in cnt] +
                    [f"{np.median(dl):.0f} [{np.quantile(dl, .25):.0f}, {np.quantile(dl, .75):.0f}]", f"{ip.mean():.2f}"])
    cap = (r"Infectious periods of infected farms by day of infectious onset. The infectious period is "
           r"$\min(\text{cull day}, \text{onset} + 4) - \text{onset}$. Cull delay is the number of days from onset "
           r"to culling, given as median [interquartile range]. The analysis window covers onsets in days (30, 50].")
    header = r"Onset day & Farms & IP $=1$ & IP $=2$ & IP $=3$ & IP $=4$ & Cull delay & Mean IP"
    write("tabS_fmd_ip_by_onset.tex", table(cap, "tab:si_fmd_ip", "lccccccc", header, rows))


def tab_mcmc():
    meta = json.load(open(MCMC_ROOT / "real_meta.json"))
    rows = [[QUANT_TEX[q], f"{d['rhat_rank']:.4f}", f"{d['ess_bulk']:.0f}", f"{d['ess_tail']:.0f}"]
            for q, d in meta["diagnostics"].items()]
    acc = [a["theta"] for a in meta["acceptance"]]
    cap = (rf"Convergence diagnostics for the MCMC analysis of the observed outbreak, based on {meta['n_chains']} chains "
           rf"of {meta['n_iter']:,} iterations with the first {meta['n_burnin']:,} discarded as burn-in. "
           r"$\hat{R}$ is the rank-normalized split-$\hat{R}$, and ESS is the bulk and tail effective sample size "
           rf"computed from all post-burn-in draws. Acceptance rates ranged from {min(acc):.2f} to {max(acc):.2f}.")
    header = r"Quantity & $\hat{R}$ & Bulk ESS & Tail ESS"
    write("tabS_fmd_mcmc.tex", table(cap, "tab:si_fmd_mcmc", "lccc", header, rows))


def tab_network():
    best = json.load(open(NPE_ROOT / "best_hparams.json"))
    meta = {m: [json.load(open(NPE_ROOT / f"seed_{s}" / f"{m}_run_meta.json")) for s in SEEDS] for m in MODELS}
    hp = {m: best[m]["best_config"] for m in MODELS}
    msd = lambda v, d: f"{np.mean(v):.{d}f} ({np.std(v, ddof=1):.{d}f})"
    lr = lambda v: rf"${v / 10 ** np.floor(np.log10(v)):.2f}\times10^{{{int(np.floor(np.log10(v)))}}}$"
    top = [
        ["Learning rate", lr(hp["cnn"]["learning_rate"]), lr(hp["gnn"]["learning_rate"])],
        ["Flow transforms", str(hp["cnn"]["num_transforms"]), str(hp["gnn"]["num_transforms"])],
        ["Flow hidden units", str(hp["cnn"]["hidden_features"]), str(hp["gnn"]["hidden_features"])],
        ["Embedding dimension", str(hp["cnn"]["emb_dim"]), str(hp["gnn"]["emb_dim"])],
        ["First / second channels", f"{hp['cnn']['channels1']} / {hp['cnn']['channels2']}", ""],
        ["GNN hidden dimension", "", str(hp["gnn"]["gnn_hidden"])],
        ["Message-passing layers", "", str(hp["gnn"]["gnn_layers"])],
        ["Nearest neighbours $k$", "", str(hp["gnn"]["gnn_k"])],
    ]
    bottom = [
        ["Embedding parameters"] + [f"{meta[m][0]['n_params_embedding']:,}" for m in MODELS],
        ["Total parameters"] + [f"{meta[m][0]['n_params_total']:,}" for m in MODELS],
        ["Best val.\\ log-prob."] + [msd([d["training"]["best_val_lp"] for d in meta[m]], 2) for m in MODELS],
        ["Epochs"] + [msd([d["training"]["epochs_run"] for d in meta[m]], 1) for m in MODELS],
    ]
    cap = (r"Network settings and training summaries for the FMD analysis. Best validation log-probability and number "
           r"of training epochs are the mean (SD) over the five training runs.")
    lines = [r"\begin{table}[h!]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:si_fmd_network}", r"\small",
             r"\begin{tabular}{lcc}", r"\toprule", r"Setting & CNN-NPE & GNN-NPE \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in top] + [r"\midrule"] + [" & ".join(r) + r" \\" for r in bottom]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tabS_fmd_network.tex", "\n".join(lines))


def tab_embedding():
    e = pd.read_csv(EVAL_ROOT / "embedding_mmd.csv")
    rows = []
    for s_ in SEEDS:
        row = [str(s_)]
        for m in MODELS:
            r = e[(e.model == m) & (e.seed == s_)].iloc[0]
            row += [f"{r.mmd_p_value:.2f}", f"{r.mahalanobis_p_value:.2f}"]
        rows.append(row)
    n_bank, n_null = int(e.n_bank.iloc[0]), int(e.n_null.iloc[0])
    cap = (rf"$p$-values of the embedding checks at the observed data for each trained network, computed against "
           rf"{n_null:,} held-out simulations with a bank of {n_bank:,} simulations from the prior.")
    lines = [r"\begin{table}[h!]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:si_fmd_embedding}", r"\small",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{CNN-NPE} & \multicolumn{2}{c}{GNN-NPE} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Network & MMD & Mahalanobis & MMD & Mahalanobis \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tabS_fmd_embedding.tex", "\n".join(lines))


def tab_local():
    s = pd.read_csv(SI_ROOT / "local_check_summary.csv")
    long = pd.read_csv(SI_ROOT / "local_check_long.csv")
    meta = json.load(open(SI_ROOT / "local_check_meta.json"))
    rows = []
    for k in ("mcmc", "cnn", "gnn"):
        for p in PARAM_NAMES:
            r = s[(s.model == k) & (s.parameter == p)].iloc[0]
            row = [LABEL[k] if p == PARAM_NAMES[0] else "", QUANT_TEX[p],
                   f"{num(r.mean_log_error)} ({r.sd_log_error:.2f})", f"{r.mean_log_width:.2f}"]
            if k == "mcmc":
                row += ["", "", ""]
            else:
                per = long[(long.model == k) & (long.parameter == p)].groupby("dataset")["log_shift_vs_mcmc"].mean()
                n_ge = int((per >= r.real_log_shift_vs_mcmc).sum())
                row += [f"{num(r.mean_log_shift_vs_mcmc)} ({r.sd_log_shift_vs_mcmc:.2f})",
                        num(r.real_log_shift_vs_mcmc), f"{n_ge}/{len(per)}"]
            rows.append(row)
    cap = (rf"Local check on {meta['n_sims']} posterior predictive datasets simulated from MCMC posterior draws. "
           r"All quantities are on the "
           r"log scale. Error is the posterior median minus the true value, given as mean (SD) over datasets and trained "
           r"networks. Width is the mean width of the 95\% credible interval. Shift is the NPE posterior median minus "
           r"the MCMC posterior median on the same dataset, given as mean (SD). Observed shift is the same quantity at "
           r"the observed outbreak, averaged over the five trained networks. The last column counts local datasets "
           r"whose network-averaged shift is at least the observed shift.")
    header = (r"Method & Parameter & Error & Width & Shift & Observed shift & Datasets $\geq$ observed")
    write("tab_fmd_local.tex", table(cap, "tab:fmd_local", "llccccc", header, rows))


def risk_map(sets, cfg):
    exp = cfg["exp_obs"].astype(np.int64)
    rem = cfg["rem_obs"].astype(np.int64)
    onset, end = infection_history(exp, rem, cfg)
    t = RISK_DAY
    inf = np.where((onset <= t) & (end > t))[0]
    sus = np.where(cfg["susceptible_mask"].astype(bool) & ~((exp >= 1) & (exp <= t)))[0]
    nxt = np.where(exp == t + 1)[0]
    lk = cfg["log_kernel"][np.ix_(inf, sus)].astype(np.float64)
    z = cfg["z"][sus].astype(np.float64)
    out = {"day": np.int32(int(cfg["t_start"]) + t), "infectious": inf, "susceptible": sus, "exposed_next": nxt}
    rows = []
    for m in ("cnn", "gnn", "mcmc"):
        th = sets[m]
        prob = np.empty((len(th), len(sus)))
        for i in range(0, len(th), RISK_CHUNK):
            c = th[i:i + RISK_CHUNK]
            pressure = np.exp(-c[:, 2][:, None, None] * lk[None, :, :]).sum(axis=1)
            prob[i:i + RISK_CHUNK] = -np.expm1(-(c[:, 0][:, None] + c[:, 1][:, None] * z[None, :]) * pressure)
        lo, med, hi = np.quantile(prob, [0.025, 0.5, 0.975], axis=0)
        out[f"{m}_q025"], out[f"{m}_median"], out[f"{m}_q975"] = lo, med, hi
        ratio = hi / lo
        rows.append({"model": m, "n_draws": len(th), "expected_exposures": float(prob.sum(axis=1).mean()),
                     "ratio_q10": float(np.quantile(ratio, 0.1)), "ratio_median": float(np.median(ratio)),
                     "ratio_q90": float(np.quantile(ratio, 0.9))})
    np.savez_compressed(PAPER_ROOT / "risk_map.npz", **out)
    df = pd.DataFrame(rows)
    df.to_csv(PAPER_ROOT / "risk_map_summary.csv", index=False)
    print(f"\nrisk map for exposure on day {int(out['day']) + 1}: {len(inf)} infectious, {len(sus)} susceptible, "
          f"{len(nxt)} exposed next day")
    print(df.round(3).to_string(index=False))


def prior_predictive(cfg):
    ref = np.load(DATA_ROOT / "reference" / "reference.npz", allow_pickle=True)
    exp = ref["exp"][:N_PRIOR_PRED].astype(np.int64)
    T = int(cfg["T"])
    daily = np.stack([(exp == t).sum(axis=1) for t in range(1, T + 1)], axis=1)
    obs = np.array([(cfg["exp_obs"] == t).sum() for t in range(1, T + 1)])
    tot = (ref["exp"] >= 1).sum(axis=1)
    q = float(np.mean(tot < obs.sum()) + 0.5 * np.mean(tot == obs.sum()))
    np.savez_compressed(PAPER_ROOT / "prior_predictive.npz", daily=daily.astype(np.int32), observed=obs.astype(np.int32),
                        total=tot.astype(np.int32), observed_total_quantile=np.float64(q))
    print(f"\nprior predictive: {len(tot)} simulations, total exposures median {np.median(tot):.0f} "
          f"[{np.quantile(tot, 0.025):.0f}, {np.quantile(tot, 0.975):.0f}], observed {obs.sum()} at quantile {q:.3f}")


def main():
    cfg = load_config()
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    sets = load_sets(cfg)
    np.savez_compressed(PAPER_ROOT / "derived_samples.npz", columns=np.array(PARAM_NAMES + ("r0", "r"), dtype=object),
                        e_ip=np.float64(mean_ip(cfg)), **sets)
    tab_posterior(sets, cfg)
    spatial_ppc(cfg, N_CPUS)
    tab_ip()
    tab_mcmc()
    tab_network()
    tab_embedding()
    tab_local()
    risk_map(sets, cfg)
    prior_predictive(cfg)
    print(f"\nSaved under {PAPER_ROOT}")


if __name__ == "__main__":
    main()
