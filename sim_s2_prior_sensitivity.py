import argparse
import contextlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from torch.distributions import Distribution, constraints

from sim_02_make_datasets import (
    simulate_stochastic_recovery,
    seed_rowsum_safe_np,
    T,
)
from sim_03_tune_npe import (
    CNN1D,
    WeightedGNN,
    DEVICE,
    SEED,
    FIXED_HPARAMS,
    set_seed,
    interp_torch,
    compute_sr_exact_torch,
    gamma_logpdf_torch,
    inf_time_to_daily,
    valid_mask_gamma,
)

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
CONFIG_ROOT = Path(os.environ.get("ILM_CONFIG_ROOT", ARC_ROOT / "configs")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()
OUT_ROOT = Path(os.environ.get("ILM_PRIOR_SENS_ROOT", ARC_ROOT / "prior_sensitivity")).expanduser().resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS_ROOT = ARC_ROOT / "strict_seeds"
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
MODELS = (("cnn", "CNN-NPE"), ("gnn", "GNN-NPE"))
METRIC_COLS = ("mae", "relative_mae_percent", "coverage_95", "ci95_width_median")
PARAM_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$"}

D_MIN = 1.001
N_POST = 3000

PARAM_NAMES = ("alpha", "beta", "gamma")

PRIOR_VARIANTS = {
    "baseline": {
        "beta_shape": 6.0,
        "beta_scale": 0.25,
        "R0_shape": 10.0,
        "R0_scale": 0.25,
        "d_mu": math.log(3) - 0.27**2 / 2,
        "d_sigma": 0.27,
        "description": "current settings",
    },
    "all_tighter": {
        "beta_shape": 12.0,
        "beta_scale": 0.125,
        "R0_shape": 20.0,
        "R0_scale": 0.125,
        "d_mu": math.log(3) - 0.15**2 / 2,
        "d_sigma": 0.15,
        "description": "all three priors tighter (sigma roughly halved)",
    },
    "bR0_tighter": {
        "beta_shape": 12.0,
        "beta_scale": 0.125,
        "R0_shape": 20.0,
        "R0_scale": 0.125,
        "d_mu": math.log(3) - 0.27**2 / 2,
        "d_sigma": 0.27,
        "description": "only beta and R0 tighter; d at baseline",
    },
    "d_tighter": {
        "beta_shape": 6.0,
        "beta_scale": 0.25,
        "R0_shape": 10.0,
        "R0_scale": 0.25,
        "d_mu": math.log(3) - 0.15**2 / 2,
        "d_sigma": 0.15,
        "description": "only d tighter",
    },
    "d_wider": {
        "beta_shape": 6.0,
        "beta_scale": 0.25,
        "R0_shape": 10.0,
        "R0_scale": 0.25,
        "d_mu": math.log(3) - 0.5**2 / 2,
        "d_sigma": 0.5,
        "description": "only d wider (sigma roughly doubled)",
    },
    "bR0_wider": {
        "beta_shape": 3.0,
        "beta_scale": 0.5,
        "R0_shape": 5.0,
        "R0_scale": 0.5,
        "d_mu": math.log(3) - 0.27**2 / 2,
        "d_sigma": 0.27,
        "description": "only beta and R0 wider (variance doubled); d at baseline",
    },
    "all_wider": {
        "beta_shape": 3.0,
        "beta_scale": 0.5,
        "R0_shape": 5.0,
        "R0_scale": 0.5,
        "d_mu": math.log(3) - 0.5**2 / 2,
        "d_sigma": 0.5,
        "description": "all three priors wider",
    },
}


def load_config():
    dat = np.load(CONFIG_ROOT / "ilm_config.npz", allow_pickle=True)
    return {
        "N": int(dat["N"]),
        "region_side": float(dat["region_side"]),
        "positions": dat["positions"].astype(np.float32),
        "log_dist": dat["log_dist"].astype(np.float32),
        "seed_idx": dat["seed_idx"].astype(np.int32),
        "lut_betas": dat["lut_betas"].astype(np.float64),
        "lut_seed_rowsums": dat["lut_seed_rowsums"].astype(np.float64),
    }


class GammaScenarioPrior(Distribution):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, cfg, beta_shape, beta_scale, R0_shape, R0_scale, d_mu, d_sigma):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([3]),
            validate_args=False,
        )
        self.beta_shape = float(beta_shape)
        self.beta_scale = float(beta_scale)
        self.R0_shape = float(R0_shape)
        self.R0_scale = float(R0_scale)
        self.d_mu = float(d_mu)
        self.d_sigma = float(d_sigma)
        self.lut_b = torch.tensor(cfg["lut_betas"], dtype=torch.float64)
        self.lut_sr = torch.tensor(cfg["lut_seed_rowsums"], dtype=torch.float64)

        ld_seed = cfg["log_dist"][cfg["seed_idx"]].astype(np.float64).copy()
        for k, sk in enumerate(np.asarray(cfg["seed_idx"]).tolist()):
            ld_seed[k, sk] = np.inf

        self.log_dist_seed = torch.tensor(ld_seed, dtype=torch.float64)
        self.lut_lo = float(cfg["lut_betas"][0])
        self.lut_hi = float(cfg["lut_betas"][-1])

    def _sr_safe(self, beta):
        b = beta.double()
        in_range = (b >= self.lut_lo) & (b <= self.lut_hi)
        sr = torch.empty_like(b)

        if in_range.any():
            sr[in_range] = interp_torch(b[in_range], self.lut_b, self.lut_sr)

        if (~in_range).any():
            sr[~in_range] = compute_sr_exact_torch(b[~in_range], self.log_dist_seed)

        return sr

    def sample(self, sample_shape=torch.Size()):
        n = max(1, int(torch.Size(sample_shape).numel()))

        b = torch._standard_gamma(
            torch.full((n,), self.beta_shape, dtype=torch.float64)
        ) * self.beta_scale

        r0 = torch._standard_gamma(
            torch.full((n,), self.R0_shape, dtype=torch.float64)
        ) * self.R0_scale

        d = torch.exp(
            self.d_mu + self.d_sigma * torch.randn(n, dtype=torch.float64)
        ).clamp_min(D_MIN)

        g = -torch.log1p(-1.0 / d)
        sr = self._sr_safe(b)
        a = r0 / (d * sr)

        theta = torch.stack([a.float(), b.float(), g.float()], dim=1)

        return theta.reshape(sample_shape + (3,)) if sample_shape else theta.squeeze(0)

    def log_prob(self, theta):
        a = theta[..., 0].double()
        b = theta[..., 1].double()
        g = theta[..., 2].double()

        ok0 = (a > 0) & (b > 0) & (g > 0)

        d = 1.0 / (1.0 - torch.exp(-g)).clamp(min=1e-12)
        sr = self._sr_safe(b)
        r0 = a * d * sr

        ok = ok0 & (d > D_MIN) & (r0 > 0)

        ld = torch.log(d.clamp(min=1e-12))

        lpd = (
            -0.5 * math.log(2 * math.pi)
            - math.log(self.d_sigma)
            - 0.5 * ((ld - self.d_mu) / self.d_sigma) ** 2
        )

        lp_b = gamma_logpdf_torch(b, self.beta_shape, self.beta_scale)
        lp_r0 = gamma_logpdf_torch(r0, self.R0_shape, self.R0_scale)

        lp = (
            lp_b
            + lp_r0
            + lpd
            + 2.0 * ld
            + torch.log(sr.clamp(min=1e-12))
            + torch.log((d - 1.0).clamp(min=1e-12))
        )

        return torch.where(ok, lp, torch.full_like(lp, -float("inf"))).float()


def draw_prior_variant(rng, variant, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    beta = float(rng.gamma(variant["beta_shape"], variant["beta_scale"]))
    r0 = float(rng.gamma(variant["R0_shape"], variant["R0_scale"]))

    log_d = rng.normal(variant["d_mu"], variant["d_sigma"])
    d = max(math.exp(log_d), D_MIN)

    gamma = -math.log(1.0 - 1.0 / d)

    sr = seed_rowsum_safe_np(
        beta,
        lut_betas,
        lut_seed_rowsums,
        log_dist,
        seed_idx,
    )

    alpha = float(r0 / (d * sr))

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "r0": r0,
        "d": d,
    }


def generate_dataset(n_keep, cfg, variant, base_seed):
    log_dist = cfg["log_dist"]
    seed_idx = cfg["seed_idx"]
    lut_betas = cfg["lut_betas"]
    lut_seed_rowsums = cfg["lut_seed_rowsums"]

    rng = np.random.default_rng(base_seed)

    thetas = np.empty((n_keep, 3), dtype=np.float32)
    xs = np.empty((n_keep, cfg["N"]), dtype=np.int16)
    stats_rows = []

    for i in range(n_keep):
        pars = draw_prior_variant(
            rng,
            variant,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
        )

        inf_t = simulate_stochastic_recovery(
            log_dist,
            pars["alpha"],
            pars["beta"],
            pars["gamma"],
            seed_idx,
            rng,
        )

        thetas[i] = [pars["alpha"], pars["beta"], pars["gamma"]]
        xs[i] = inf_t.astype(np.int16)

        n_inf = int((inf_t >= 0).sum())
        active = int(inf_t[inf_t >= 0].max()) if n_inf > 0 else 0

        stats_rows.append({
            "alpha": pars["alpha"],
            "beta": pars["beta"],
            "gamma": pars["gamma"],
            "r0": pars["r0"],
            "d": pars["d"],
            "attack_rate": n_inf / cfg["N"],
            "active_days": active,
        })

        if (i + 1) % 1000 == 0:
            print(f"    generated {i + 1}/{n_keep}", flush=True)

    return thetas, xs, pd.DataFrame(stats_rows)


def encode_cnn(x, T_len, mean=None, std=None):
    c = np.log1p(inf_time_to_daily(x.astype(np.int16), T_len))

    if mean is None:
        mean = c.mean(axis=0, keepdims=True)
        std = c.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)

    x_enc = ((c - mean) / std)[:, None, :].astype(np.float32)

    return x_enc, mean, std


def encode_gnn(x, T_len):
    s = -1.0 / T_len
    return np.where(x >= 0, x.astype(np.float32) / T_len, s).astype(np.float32)


def sample_valid_posterior_strict(posterior, x_obs, cfg, n_post):
    batch_draws = max(4000, 4 * n_post)
    kept = []
    tries = 0
    max_tries = 50

    while sum(arr.shape[0] for arr in kept) < n_post and tries < max_tries:
        tries += 1

        raw = posterior.sample(
            (batch_draws,),
            x=x_obs,
            show_progress_bars=False,
        ).detach().cpu().numpy()

        mask = valid_mask_gamma(
            raw,
            cfg["lut_betas"],
            cfg["lut_seed_rowsums"],
            cfg["log_dist"],
            cfg["seed_idx"],
        )

        good = raw[mask]

        if good.shape[0] > 0:
            kept.append(good.astype(np.float32))

    n_kept = sum(arr.shape[0] for arr in kept)

    if n_kept < n_post:
        raise RuntimeError(
            f"Only {n_kept}/{n_post} valid posterior samples after {tries} tries."
        )

    return np.concatenate(kept, axis=0)[:n_post]


def train_npe(theta, x, prior, emb, hps):
    set_seed(hps.get("seed", SEED))

    de_fn = posterior_nn(
        model=hps["flow_model"],
        embedding_net=emb,
        hidden_features=hps["hidden_features"],
        num_transforms=hps["num_transforms"],
    )

    inf = SNPE(prior=prior, density_estimator=de_fn, device=DEVICE)

    inf.append_simulations(
        torch.tensor(theta, dtype=torch.float32, device=DEVICE),
        torch.tensor(x, dtype=torch.float32, device=DEVICE),
    )

    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        de = inf.train(
            max_num_epochs=hps["max_num_epochs"],
            training_batch_size=hps["batch_size"],
            learning_rate=hps["learning_rate"],
            validation_fraction=hps["validation_fraction"],
            stop_after_epochs=hps["stop_after_epochs"],
            show_train_summary=False,
        )

    train_sec = time.perf_counter() - t0

    info = {"train_sec": float(train_sec)}

    try:
        val_log_probs = inf.summary.get("validation_log_probs", [])
        if val_log_probs:
            info["epochs_run"] = int(len(val_log_probs))
            info["best_epoch"] = int(np.argmax(val_log_probs))
            info["best_val_lp"] = float(np.max(val_log_probs))
    except Exception:
        pass

    return inf.build_posterior(de), info


def run_inference_test(posterior, x_te, cfg, label):
    n = x_te.shape[0]
    samples = np.empty((n, N_POST, 3), dtype=np.float32)

    t0 = time.perf_counter()

    for i in range(n):
        x_obs = torch.tensor(x_te[i:i + 1], dtype=torch.float32, device=DEVICE)
        samples[i] = sample_valid_posterior_strict(posterior, x_obs, cfg, N_POST)

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"    [{label}] {i + 1}/{n}  {elapsed:.0f}s  "
                f"({elapsed / (i + 1) * 1000:.0f}ms/obs)",
                flush=True,
            )

    return samples, float(time.perf_counter() - t0)


def baseline_dir(seed):
    return SEEDS_ROOT / f"seed_{seed}" / "gamma" / "final"


def atomic_savez(path, **arrays):
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_or_generate_data(name, variant, cfg, n_train, n_test, seed_offset):
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_path = out_dir / "train.npz"
    te_path = out_dir / "test.npz"
    cfg_path = out_dir / "prior_config.json"

    if tr_path.exists() and te_path.exists() and cfg_path.exists():
        with open(cfg_path) as f:
            saved = json.load(f)
        if saved.get("variant") != variant:
            raise ValueError(f"Saved prior settings for {name} differ from the current code; remove {out_dir} to regenerate.")
        tr = np.load(tr_path, allow_pickle=True)
        te = np.load(te_path, allow_pickle=True)
        if tr["theta"].shape[0] != n_train or te["theta"].shape[0] != n_test:
            raise ValueError(f"Saved data sizes for {name} differ from n_train={n_train}, n_test={n_test}.")
        print(f"  reusing data in {out_dir}", flush=True)
        return tr["theta"].astype(np.float32), tr["x"], te["theta"].astype(np.float32), te["x"]

    print("  generating train and test data", flush=True)
    t0 = time.perf_counter()
    th_tr, x_tr, stats_tr = generate_dataset(n_train, cfg, variant, SEED + seed_offset)
    th_te, x_te, stats_te = generate_dataset(n_test, cfg, variant, SEED + seed_offset + 1)
    print(f"  data: train={th_tr.shape}, test={th_te.shape} ({time.perf_counter() - t0:.0f}s)", flush=True)

    for path, th, x in ((tr_path, th_tr, x_tr), (te_path, th_te, x_te)):
        atomic_savez(path, theta=th, x=x, T=np.int32(T), param_names=np.array(PARAM_NAMES, dtype=object))
    atomic_write_text(out_dir / "train_stats.csv", stats_tr.to_csv(index=False))
    atomic_write_text(out_dir / "test_stats.csv", stats_te.to_csv(index=False))
    atomic_write_text(cfg_path, json.dumps({
        "name": name,
        "variant": variant,
        "data_seed_train": SEED + seed_offset,
        "data_seed_test": SEED + seed_offset + 1,
    }, indent=2))

    return th_tr, x_tr, th_te, x_te


def model_hparams(tag, saved, seed):
    best = saved[tag]["best_config"]
    hps = {
        **FIXED_HPARAMS,
        "seed": int(seed),
        "learning_rate": float(best["learning_rate"]),
        "num_transforms": int(best["num_transforms"]),
        "emb_dim": int(best["emb_dim"]),
        "hidden_features": int(best.get("hidden_features", 64)),
    }
    if tag == "cnn":
        hps["channels1"] = int(best["channels1"])
        hps["channels2"] = int(best["channels2"])
    else:
        hps["gnn_hidden"] = int(best["gnn_hidden"])
        hps["gnn_layers"] = int(best["gnn_layers"])
        hps["gnn_k"] = int(best["gnn_k"])
    return hps


def build_embedding(tag, hps, cfg, T_len):
    if tag == "cnn":
        return CNN1D(
            T_len,
            emb_dim=hps["emb_dim"],
            channels1=hps["channels1"],
            channels2=hps["channels2"],
            pool_out=hps["pool_out"],
            dropout=hps["dropout"],
        ).to(DEVICE)
    return WeightedGNN(
        cfg["positions"],
        cfg["region_side"],
        T_len,
        emb_dim=hps["emb_dim"],
        hidden=hps["gnn_hidden"],
        n_layers=hps["gnn_layers"],
        k=hps["gnn_k"],
        dropout=hps["dropout"],
    ).to(DEVICE)


def run_variant_seed(name, variant, cfg, saved_hparams, data, seed):
    seed_dir = OUT_ROOT / name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    th_tr, x_tr, th_te, x_te = data
    T_len = T

    prior = GammaScenarioPrior(
        cfg,
        variant["beta_shape"],
        variant["beta_scale"],
        variant["R0_shape"],
        variant["R0_scale"],
        variant["d_mu"],
        variant["d_sigma"],
    )

    x_cnn_tr, cnn_mean, cnn_std = encode_cnn(x_tr, T_len)
    x_cnn_te, _, _ = encode_cnn(x_te, T_len, cnn_mean, cnn_std)
    encoded = {
        "cnn": (x_cnn_tr, x_cnn_te),
        "gnn": (encode_gnn(x_tr, T_len), encode_gnn(x_te, T_len)),
    }

    for tag, label in MODELS:
        npz_path = seed_dir / f"{tag}_posterior_samples.npz"
        meta_path = seed_dir / f"{tag}_meta.json"
        if npz_path.exists() and meta_path.exists():
            print(f"  [skip] {name} seed={seed} {label}", flush=True)
            continue

        hps = model_hparams(tag, saved_hparams, seed)
        set_seed(seed)
        emb = build_embedding(tag, hps, cfg, T_len)
        n_emb = int(sum(p.numel() for p in emb.parameters() if p.requires_grad))

        print(f"  training {label}  seed={seed}  embedding params={n_emb}", flush=True)
        post, info = train_npe(th_tr, encoded[tag][0], prior, emb, hps)
        print(
            f"    train done: {info['train_sec']:.0f}s "
            f"epochs={info.get('epochs_run', '?')} "
            f"best_epoch={info.get('best_epoch', '?')}",
            flush=True,
        )

        set_seed(seed)
        samp, isec = run_inference_test(post, encoded[tag][1], cfg, label)

        np.savez_compressed(
            npz_path,
            samples=samp,
            theta=th_te,
            param_names=np.array(PARAM_NAMES, dtype=object),
            n_post_samples=np.int32(N_POST),
        )

        meta = {
            "model": label,
            "variant": name,
            "seed": int(seed),
            "n_post_samples": N_POST,
            "n_params_embedding": n_emb,
            "training": {**info, "n_train": int(th_tr.shape[0])},
            "infer_sec": isec,
            "hparams": hps,
        }
        if tag == "gnn":
            meta["learned_lambda_per_layer"] = emb.lambdas()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"    saved {npz_path}", flush=True)


def compute_metrics(samples, theta_true):
    rows = []
    for j, pname in enumerate(PARAM_NAMES):
        true = theta_true[:, j].astype(np.float64)
        abs_err = np.abs(np.median(samples[:, :, j], axis=1) - true)
        lo = np.quantile(samples[:, :, j], 0.025, axis=1)
        hi = np.quantile(samples[:, :, j], 0.975, axis=1)
        rows.append({
            "parameter": pname,
            "mae": float(abs_err.mean()),
            "relative_mae_percent": float(100.0 * abs_err.mean() / true.mean()),
            "coverage_95": float(((true >= lo) & (true <= hi)).mean()),
            "ci95_width_median": float(np.median(hi - lo)),
        })
    return rows


def load_samples(name, tag, seed):
    if name == "baseline":
        d = baseline_dir(seed)
        npz_path = d / f"{tag}_posterior_samples.npz"
        theta_path = d / "theta_test.npy"
        if not (npz_path.exists() and theta_path.exists()):
            return None
        samples = np.load(npz_path, allow_pickle=True)["samples"].astype(np.float32)
        return samples, np.load(theta_path).astype(np.float32)
    npz_path = OUT_ROOT / name / f"seed_{seed}" / f"{tag}_posterior_samples.npz"
    if not npz_path.exists():
        return None
    z = np.load(npz_path, allow_pickle=True)
    return z["samples"].astype(np.float32), z["theta"].astype(np.float32)


def aggregate_summary(seeds):
    print(f"\n{'=' * 70}\nAggregating sensitivity summary\n{'=' * 70}", flush=True)

    rows = []
    missing = []
    for name in PRIOR_VARIANTS:
        for seed in seeds:
            for tag, label in MODELS:
                loaded = load_samples(name, tag, seed)
                if loaded is None:
                    missing.append(f"{name}/seed_{seed}/{tag}")
                    continue
                samples, theta = loaded
                for r in compute_metrics(samples, theta):
                    rows.append({"variant": name, "model": label, "seed": int(seed), **r})

    if missing:
        print(f"  missing {len(missing)} run(s): {', '.join(missing)}", flush=True)
    if not rows:
        print("  no results to aggregate", flush=True)
        return None

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_ROOT / "sensitivity_per_seed.csv", index=False)

    out = []
    for (name, label, pname), g in per_seed.groupby(["variant", "model", "parameter"], sort=False):
        row = {"variant": name, "model": label, "parameter": pname, "n_seeds": int(len(g))}
        for c in METRIC_COLS:
            v = g[c].to_numpy(float)
            row[f"{c}_mean"] = float(v.mean())
            row[f"{c}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(OUT_ROOT / "sensitivity_summary.csv", index=False)

    write_table(summary)
    print(summary.round(3).to_string(index=False))
    print(f"\n  saved: {OUT_ROOT}", flush=True)
    return summary


def fmt(mean, sd, digits):
    return f"{mean:.{digits}f}" if not np.isfinite(sd) else f"{mean:.{digits}f} ({sd:.{digits}f})"


def write_table(summary):
    labels = {name: f"Prior {k + 1}" for k, name in enumerate(PRIOR_VARIANTS)}
    body = []
    for k, name in enumerate(PRIOR_VARIANTS):
        sub = summary[summary["variant"] == name]
        if sub.empty:
            continue
        first = True
        for _, label in MODELS:
            g = sub[sub["model"] == label]
            for i, (_, r) in enumerate(g.iterrows()):
                body.append(" & ".join([
                    labels[name] if first else "", label if i == 0 else "", PARAM_TEX[r["parameter"]],
                    fmt(r["mae_mean"], r["mae_sd"], 3), fmt(r["relative_mae_percent_mean"], r["relative_mae_percent_sd"], 1),
                    fmt(r["ci95_width_median_mean"], r["ci95_width_median_sd"], 3),
                    fmt(r["coverage_95_mean"], r["coverage_95_sd"], 3),
                ]) + r" \\")
                first = False
        if k < len(PRIOR_VARIANTS) - 1:
            body.append(r"\midrule")
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\caption{Prior sensitivity analysis for the stochastic-removal scenario. Entries are mean (SD) over five "
        r"independently trained networks. Relative MAE is $100 \times \mathrm{MAE}/\bar{\theta}$, where "
        r"$\bar{\theta}$ is the mean true value of the parameter in the test set of each prior. Width is the median "
        r"width of the 95\% credible interval across test epidemics.}",
        r"\label{tab:si_prior_sensitivity}", r"\small",
        r"\begin{tabular}{lllcccc}", r"\toprule",
        r"Prior & Model & Parameter & MAE & Relative MAE (\%) & Width & Coverage \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    (OUT_ROOT / "sensitivity_table.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--skip_run", action="store_true")

    args = parser.parse_args()

    if args.variants == "all":
        names = list(PRIOR_VARIANTS.keys())
    else:
        names = [x.strip() for x in args.variants.split(",")]
        bad = [x for x in names if x not in PRIOR_VARIANTS]

        if bad:
            raise ValueError(f"Unknown variants: {bad}. Choices: {list(PRIOR_VARIANTS)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.skip_run:
        aggregate_summary(DEFAULT_SEEDS)
        print(f"\nAll outputs in {OUT_ROOT}", flush=True)
        return

    cfg = load_config()

    print(f"device={DEVICE}  N={cfg['N']}  T={T}  N_POST={N_POST}", flush=True)
    print(f"variants: {names}  seeds: {args.seeds}", flush=True)

    hp_path = STRICT_ROOT / "gamma" / "best_hparams.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"Missing gamma hparams: {hp_path}")
    with open(hp_path) as f:
        saved_hparams = json.load(f)

    seed_offsets = {name: 1000 + 100 * i for i, name in enumerate(PRIOR_VARIANTS.keys())}
    t_start = time.perf_counter()

    for name in names:
        variant = PRIOR_VARIANTS[name]
        print(f"\n{'=' * 70}\n[{name}] {variant['description']}\n{'=' * 70}", flush=True)

        if name == "baseline":
            for seed in args.seeds:
                d = baseline_dir(seed)
                ok = all((d / f).exists() for f in ("cnn_posterior_samples.npz", "gnn_posterior_samples.npz", "theta_test.npy"))
                print(f"  seed={seed}: {'found main results' if ok else 'MISSING main results, run sim_04 first'} in {d}", flush=True)
            continue

        data = load_or_generate_data(name, variant, cfg, args.n_train, args.n_test, seed_offsets[name])
        for seed in args.seeds:
            run_variant_seed(name, variant, cfg, saved_hparams, data, seed)

    print(f"\nTotal wall time: {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    print("Run with --skip_run after all seeds finish to aggregate.", flush=True)


if __name__ == "__main__":
    main()
