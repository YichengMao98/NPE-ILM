import argparse
import contextlib
import io
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from torch.distributions import Distribution, constraints

from fmd_02_make_datasets import (
    BETA_SCALE,
    BETA_SHAPE,
    PARAM_NAMES,
    R0_SCALE,
    R0_SHAPE,
    load_config,
    mean_ip,
)
from sim_03_tune_npe import SAGELayer, WeightedSAGELayer

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("FMD_OUT_ROOT", BASE_DIR / "output_fmd")).expanduser().resolve()
DATA_ROOT = OUT_ROOT / "data"
NPE_ROOT = OUT_ROOT / "npe"

N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))
os.environ["OMP_NUM_THREADS"] = str(N_CPUS)
os.environ["MKL_NUM_THREADS"] = str(N_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPUS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(N_CPUS)
    try:
        torch.set_num_interop_threads(max(1, min(4, N_CPUS)))
    except RuntimeError:
        pass

N_PARAMS = len(PARAM_NAMES)
N_TRIALS = 20
SEED = 1234
N_EVAL_POST = 500

FIXED_HPARAMS = {
    "flow_model": "maf",
    "hidden_features": 64,
    "pool_out": 8,
    "dropout": 0.05,
    "max_num_epochs": 200,
    "batch_size": 64,
    "validation_fraction": 0.1,
    "stop_after_epochs": 20,
}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interp_torch(x, xp, fp):
    xc = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, xc).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = (xc - x0) / (x1 - x0).clamp(min=1e-12)
    return f0 + t * (f1 - f0)


def gamma_logpdf_torch(x, shape, scale):
    x = x.double()
    return (shape - 1.0) * torch.log(x.clamp(min=1e-300)) - x / scale - shape * math.log(scale) - math.lgamma(shape)


class FMDPrior(Distribution):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, cfg):
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([N_PARAMS]), validate_args=False)
        self.e_ip = mean_ip(cfg)
        self.lut_b = torch.tensor(cfg["lut_betas"], dtype=torch.float64)
        self.lut_all = torch.tensor(cfg["lut_all"], dtype=torch.float64)
        self.lut_z1 = torch.tensor(cfg["lut_z1"], dtype=torch.float64)
        sus = np.where(cfg["susceptible_mask"])[0]
        self.lk = torch.tensor(cfg["log_kernel"][np.ix_(cfg["active_idx"], sus)].astype(np.float64))
        self.z1 = torch.tensor(cfg["z"][sus] == 1)

    def lambda0(self, beta, r):
        b = beta.double()
        l_all = torch.empty_like(b)
        l_z1 = torch.empty_like(b)
        inside = (b >= self.lut_b[0]) & (b <= self.lut_b[-1])
        if inside.any():
            l_all[inside] = interp_torch(b[inside], self.lut_b, self.lut_all)
            l_z1[inside] = interp_torch(b[inside], self.lut_b, self.lut_z1)
        if (~inside).any():
            k = torch.exp(-b[~inside][:, None, None] * self.lk[None])
            l_all[~inside] = k.sum(dim=2).mean(dim=1)
            l_z1[~inside] = k[:, :, self.z1].sum(dim=2).mean(dim=1)
        return r * l_all + (1.0 - r) * l_z1

    def sample(self, sample_shape=torch.Size()):
        n = max(1, int(torch.Size(sample_shape).numel()))
        b = torch.distributions.Gamma(BETA_SHAPE, 1.0 / BETA_SCALE).sample((n,)).double()
        r0 = torch.distributions.Gamma(R0_SHAPE, 1.0 / R0_SCALE).sample((n,)).double()
        r = torch.rand(n, dtype=torch.float64)
        a = r0 / (self.e_ip * self.lambda0(b, r))
        t = torch.stack([r * a, (1.0 - r) * a, b], dim=1).float()
        return t.reshape(torch.Size(sample_shape) + (N_PARAMS,)) if sample_shape else t.squeeze(0)

    def log_prob(self, theta):
        a0 = theta[..., 0].double()
        a1 = theta[..., 1].double()
        b = theta[..., 2].double()
        ok = (a0 > 0) & (a1 > 0) & (b > 0)
        a = (a0 + a1).clamp(min=1e-300)
        r = (a0 / a).clamp(0.0, 1.0)
        scale = self.e_ip * self.lambda0(b.clamp(min=1e-6), r)
        lp = (
            gamma_logpdf_torch(b, BETA_SHAPE, BETA_SCALE)
            + gamma_logpdf_torch(a * scale, R0_SHAPE, R0_SCALE)
            + torch.log(scale.clamp(min=1e-300))
            - torch.log(a)
        )
        return torch.where(ok, lp, torch.full_like(lp, -float("inf"))).float()


def valid_mask(samples):
    s = np.asarray(samples, dtype=np.float64)
    return np.isfinite(s).all(axis=1) & (s > 0).all(axis=1)


def full_history(exp, rem, cfg):
    T = int(cfg["T"])
    exp = np.array(exp, dtype=np.int16, ndmin=2)
    rem = np.array(rem, dtype=np.int16, ndmin=2)
    flag = exp >= 1
    has_rem = rem >= 0
    exp_full = np.where(flag, exp, 0).astype(np.int16)
    rem_full = np.where(has_rem, np.minimum(rem, T), 0).astype(np.int16)
    for key in ("active", "removed"):
        idx = cfg[f"{key}_idx"]
        flag[:, idx] = True
        has_rem[:, idx] = True
        exp_full[:, idx] = cfg[f"{key}_exp"]
        rem_full[:, idx] = np.minimum(cfg[f"{key}_rem"], T)
    idx = cfg["exposed_idx"]
    flag[:, idx] = True
    exp_full[:, idx] = cfg["exposed_exp"]
    return flag, exp_full, has_rem, rem_full


def encode_gnn(exp, rem, cfg):
    T = float(cfg["T"])
    flag, exp_full, has_rem, rem_full = full_history(exp, rem, cfg)
    parts = [
        flag.astype(np.float32),
        np.where(flag, exp_full / T, 0.0).astype(np.float32),
        has_rem.astype(np.float32),
        np.where(has_rem, rem_full / T, 0.0).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def daily_counts(exp, rem, cfg):
    T = int(cfg["T"])
    flag, exp_full, has_rem, rem_full = full_history(exp, rem, cfg)
    z1 = cfg["z"].astype(bool)
    out = np.zeros((exp_full.shape[0], 3, T), dtype=np.float32)
    for t in range(1, T + 1):
        new = flag & (exp_full == t)
        out[:, 0, t - 1] = (new & ~z1).sum(axis=1)
        out[:, 1, t - 1] = (new & z1).sum(axis=1)
        if t < T:
            out[:, 2, t - 1] = (has_rem & (rem_full == t)).sum(axis=1)
    return out


def cnn_norm_stats(exp, rem, cfg):
    c = np.log1p(daily_counts(exp, rem, cfg))
    mean = c.mean(axis=0, keepdims=True)
    std = c.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def encode_cnn(exp, rem, cfg, mean, std):
    c = np.log1p(daily_counts(exp, rem, cfg))
    return ((c - mean) / std).astype(np.float32)


def load_split(path):
    d = np.load(path, allow_pickle=True)
    return d["theta"].astype(np.float32), d["exp"].astype(np.int16), d["rem"].astype(np.int16)


class CNN1D(nn.Module):
    def __init__(self, in_channels=3, emb_dim=64, channels1=32, channels2=128, pool_out=8, dropout=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels1, channels2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(pool_out),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels2 * pool_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def knn_graph(positions, k):
    dists = squareform(pdist(positions, "euclidean"))
    np.fill_diagonal(dists, np.inf)
    knn = np.argsort(dists, axis=1)[:, :k].astype(np.int64)
    knn_d = np.maximum(np.take_along_axis(dists, knn, axis=1), 1e-6)
    return knn, knn_d


class FMDGraphBase(nn.Module):
    def __init__(self, positions, region_side, z, hidden, emb_dim, dropout):
        super().__init__()
        self.n_nodes = int(positions.shape[0])
        self.register_buffer("pos_n", torch.tensor(positions / region_side, dtype=torch.float32))
        self.register_buffer("z", torch.tensor(np.asarray(z, dtype=np.float32)[:, None]))
        self.enc = nn.Linear(7, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def node_features(self, x):
        B = x.shape[0]
        x = x.reshape(B, 4, self.n_nodes).permute(0, 2, 1)
        pos = self.pos_n.unsqueeze(0).expand(B, -1, -1)
        z = self.z.unsqueeze(0).expand(B, -1, -1)
        return F.relu(self.enc(torch.cat([x, z, pos], -1)))


class GNN(FMDGraphBase):
    def __init__(self, positions, region_side, z, emb_dim=32, hidden=64, n_layers=3, k=5, dropout=0.05):
        super().__init__(positions, region_side, z, hidden, emb_dim, dropout)
        knn, _ = knn_graph(positions, k)
        self.register_buffer("knn", torch.tensor(knn, dtype=torch.long))
        self.lyrs = nn.ModuleList([SAGELayer(hidden, hidden, dropout) for _ in range(n_layers)])

    def forward(self, x):
        h = self.node_features(x)
        for layer in self.lyrs:
            h = layer(h, self.knn)
        return self.head(h.mean(1))


class WeightedGNN(FMDGraphBase):
    def __init__(self, positions, region_side, z, emb_dim=32, hidden=64, n_layers=3, k=5, dropout=0.05):
        super().__init__(positions, region_side, z, hidden, emb_dim, dropout)
        knn, knn_d = knn_graph(positions, k)
        self.register_buffer("knn", torch.tensor(knn, dtype=torch.long))
        self.register_buffer("knn_logd", torch.tensor(np.log(knn_d / np.median(knn_d)), dtype=torch.float32))
        self.lyrs = nn.ModuleList([WeightedSAGELayer(hidden, hidden, dropout) for _ in range(n_layers)])

    def forward(self, x):
        h = self.node_features(x)
        for layer in self.lyrs:
            h = layer(h, self.knn, self.knn_logd)
        return self.head(h.mean(1))

    def lambdas(self):
        return [float(layer.lam().detach().cpu()) for layer in self.lyrs]


def build_cnn(hps):
    return CNN1D(
        in_channels=3,
        emb_dim=hps["emb_dim"],
        channels1=hps["channels1"],
        channels2=hps["channels2"],
        pool_out=hps["pool_out"],
        dropout=hps["dropout"],
    )


def build_gnn(hps, cfg, weighted):
    cls = WeightedGNN if weighted else GNN
    return cls(
        cfg["positions_local"].astype(np.float64),
        float(cfg["region_side"]),
        cfg["z"],
        emb_dim=hps["emb_dim"],
        hidden=hps["gnn_hidden"],
        n_layers=hps["gnn_layers"],
        k=hps["gnn_k"],
        dropout=hps["dropout"],
    )


def train_npe(th_tr, x_tr, prior, emb, hps, device=DEVICE):
    set_seed(hps.get("seed", SEED))
    de_fn = posterior_nn(
        model=hps["flow_model"],
        embedding_net=emb,
        hidden_features=hps["hidden_features"],
        num_transforms=hps["num_transforms"],
        z_score_x="none",
    )
    inf = SNPE(prior=prior, density_estimator=de_fn, device=device)
    inf.append_simulations(
        torch.tensor(th_tr, dtype=torch.float32, device=device),
        torch.tensor(x_tr, dtype=torch.float32, device=device),
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
    info = {"train_sec": float(time.perf_counter() - t0)}
    vl = inf.summary.get("validation_log_probs", [])
    if vl:
        info["epochs_run"] = int(len(vl))
        info["best_epoch"] = int(np.argmax(vl))
        info["best_val_lp"] = float(np.max(vl))
    return inf.build_posterior(de), de, info


def sample_valid(posterior, x_obs, n_post, max_tries=50):
    batch = max(4000, 4 * n_post)
    kept = []
    for _ in range(max_tries):
        raw = posterior.sample((batch,), x=x_obs, show_progress_bars=False).detach().cpu().numpy()
        good = raw[valid_mask(raw)]
        if good.shape[0]:
            kept.append(good.astype(np.float32))
        if sum(k.shape[0] for k in kept) >= n_post:
            return np.concatenate(kept, axis=0)[:n_post]
    return None


def evaluate(posterior, x_te, th_te, scale, device=DEVICE):
    n = th_te.shape[0]
    med = np.full((n, N_PARAMS), np.nan, dtype=np.float64)
    for i in range(n):
        xo = torch.tensor(x_te[i:i + 1], dtype=torch.float32, device=device)
        draw = sample_valid(posterior, xo, N_EVAL_POST)
        if draw is not None:
            med[i] = np.median(draw, axis=0)
    ok = np.isfinite(med).all(axis=1)
    if not ok.any():
        return float("inf"), 0, [float("nan")] * N_PARAMS
    err = np.abs(med[ok] - th_te[ok]) / scale
    return float(err.mean()), int(ok.sum()), err.mean(axis=0).tolist()


def make_objective(kind, th_tr, x_tr, th_te, x_te, prior, cfg, scale):
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        nt = trial.suggest_int("num_transforms", 3, 10)
        if kind == "cnn":
            ed = trial.suggest_categorical("emb_dim", [32, 64, 128, 256])
            c1 = trial.suggest_categorical("channels1", [16, 32, 64])
            c2 = trial.suggest_categorical("channels2", [64, 128, 256])
            hf = trial.suggest_categorical("hidden_features", [32, 64, 128])
            hps = {**FIXED_HPARAMS, "learning_rate": lr, "num_transforms": nt, "emb_dim": ed,
                   "channels1": c1, "channels2": c2, "hidden_features": hf, "seed": SEED + trial.number}
            emb = build_cnn(hps)
        else:
            ed = trial.suggest_categorical("emb_dim", [16, 32, 64, 128])
            gh = trial.suggest_categorical("gnn_hidden", [32, 64, 128])
            gl = trial.suggest_int("gnn_layers", 2, 5)
            gk = trial.suggest_categorical("gnn_k", [5, 10, 15, 20])
            hf = trial.suggest_categorical("hidden_features", [32, 64, 128])
            hps = {**FIXED_HPARAMS, "learning_rate": lr, "num_transforms": nt, "emb_dim": ed,
                   "gnn_hidden": gh, "gnn_layers": gl, "gnn_k": gk, "hidden_features": hf,
                   "seed": SEED + trial.number}
            emb = build_gnn(hps, cfg, weighted=False)
        set_seed(hps["seed"])
        posterior, _, info = train_npe(th_tr, x_tr, prior, emb.to(DEVICE), hps)
        score, n_ok, per_param = evaluate(posterior, x_te, th_te, scale)
        trial.set_user_attr("val_log_prob", info.get("best_val_lp", float("nan")))
        trial.set_user_attr("train_sec", info["train_sec"])
        trial.set_user_attr("valid_obs", n_ok)
        trial.set_user_attr("n_epochs", info.get("epochs_run", 0))
        trial.set_user_attr("best_epoch", info.get("best_epoch", -1))
        trial.set_user_attr("scaled_mae_per_param", per_param)
        print(
            f"  {kind.upper()} {trial.number:3d}  {trial.params}  scaled_mae={score:.4f}  "
            f"valid={n_ok}/{len(th_te)}  epochs={info.get('epochs_run', 0)}  t={info['train_sec']:.0f}s",
            flush=True,
        )
        return score

    return objective


def run_study(objective, n_trials):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    b = study.best_trial
    return {
        "best_config": b.params,
        "best_scaled_mae": float(b.value),
        "best_scaled_mae_per_param": b.user_attrs.get("scaled_mae_per_param"),
        "best_val_lp": float(b.user_attrs.get("val_log_prob", float("nan"))),
        "best_n_epochs": int(b.user_attrs.get("n_epochs", 0)),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "scaled_mae": float(t.value),
                "scaled_mae_per_param": t.user_attrs.get("scaled_mae_per_param"),
                "val_lp": float(t.user_attrs.get("val_log_prob", float("nan"))),
                "valid_obs": int(t.user_attrs.get("valid_obs", 0)),
                "n_epochs": int(t.user_attrs.get("n_epochs", 0)),
                "train_sec": float(t.user_attrs.get("train_sec", float("nan"))),
            }
            for t in study.trials
            if t.value is not None
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["cnn", "gnn"], choices=["cnn", "gnn"])
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    cfg = load_config()
    prior = FMDPrior(cfg)
    pilot = DATA_ROOT / "pilot"
    th_tr, exp_tr, rem_tr = load_split(pilot / "train.npz")
    th_te, exp_te, rem_te = load_split(pilot / "test.npz")
    scale = th_tr.astype(np.float64).std(axis=0)

    NPE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = NPE_ROOT / "best_hparams.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    out.update({
        "param_names": list(PARAM_NAMES),
        "objective": "mean absolute error of posterior medians, each parameter divided by its pilot-train standard deviation",
        "objective_scale": scale.tolist(),
        "n_pilot_train": int(th_tr.shape[0]),
        "n_pilot_test": int(th_te.shape[0]),
        "n_eval_post": N_EVAL_POST,
        "fixed_hparams": FIXED_HPARAMS,
        "z_score_x": "none",
    })

    print(f"device={DEVICE}  threads={N_CPUS}  n_trials={args.n_trials}  pilot train={th_tr.shape[0]} test={th_te.shape[0]}")

    if "cnn" in args.models:
        mean, std = cnn_norm_stats(exp_tr, rem_tr, cfg)
        x_tr = encode_cnn(exp_tr, rem_tr, cfg, mean, std)
        x_te = encode_cnn(exp_te, rem_te, cfg, mean, std)
        print("\n=== CNN tuning ===", flush=True)
        t0 = time.perf_counter()
        res = run_study(make_objective("cnn", th_tr, x_tr, th_te, x_te, prior, cfg, scale), args.n_trials)
        out["cnn"] = {**res, "tuning_wall_min": (time.perf_counter() - t0) / 60}
        print(f"CNN best scaled_mae={res['best_scaled_mae']:.4f}  {res['best_config']}", flush=True)
        json.dump(out, open(out_path, "w"), indent=2)

    if "gnn" in args.models:
        x_tr = encode_gnn(exp_tr, rem_tr, cfg)
        x_te = encode_gnn(exp_te, rem_te, cfg)
        print("\n=== GNN tuning ===", flush=True)
        t0 = time.perf_counter()
        res = run_study(make_objective("gnn", th_tr, x_tr, th_te, x_te, prior, cfg, scale), args.n_trials)
        out["gnn"] = {**res, "tuning_wall_min": (time.perf_counter() - t0) / 60}
        print(f"GNN best scaled_mae={res['best_scaled_mae']:.4f}  {res['best_config']}", flush=True)
        json.dump(out, open(out_path, "w"), indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
