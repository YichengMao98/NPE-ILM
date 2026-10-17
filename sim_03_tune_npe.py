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

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
CONFIG_ROOT = Path(os.environ.get("ILM_CONFIG_ROOT", ARC_ROOT / "configs")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()

N_CPUS = max(1, int(os.environ.get("ILM_N_CPUS", os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
os.environ["OMP_NUM_THREADS"] = str(N_CPUS)
os.environ["MKL_NUM_THREADS"] = str(N_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPUS)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "cpu"
    print("MPS detected; sbi requires CPU")
else:
    DEVICE = "cpu"
    torch.set_num_threads(N_CPUS)
    try:
        torch.set_num_interop_threads(max(1, min(4, N_CPUS)))
    except RuntimeError:
        pass

IP_FIXED = 3

BETA_SHAPE = 6.0
BETA_SCALE = 1.0 / 4.0
R0_SHAPE = 10.0
R0_SCALE = 1.0 / 4.0

MU_D = math.log(3) - 0.27**2 / 2
SIGMA_D = 0.27
D_MIN = 1.001

RHO_EPS = 1e-6

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
    xp = xp.to(x.device)
    fp = fp.to(x.device)
    xc = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, xc).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = (xc - x0) / (x1 - x0).clamp(min=1e-12)
    return f0 + t * (f1 - f0)


def compute_sr_exact_torch(betas, log_dist_seed):
    n = betas.shape[0]
    chunk = 500
    out = torch.empty(n, dtype=torch.float64, device=betas.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        bs = betas[s:e]
        W = torch.exp(-bs[:, None, None] * log_dist_seed[None, :, :])
        out[s:e] = W.sum(dim=2).mean(dim=1)
    return out


def compute_sr_exact_np(betas, log_dist, seed_idx):
    betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    ld_seed = log_dist[seed_idx].astype(np.float64).copy()
    for k, sk in enumerate(seed_idx):
        ld_seed[k, sk] = np.inf
    n = len(betas)
    out = np.empty(n, dtype=np.float64)
    chunk = 500
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        bs = betas[s:e]
        W = np.exp(-bs[:, None, None] * ld_seed[None, :, :])
        out[s:e] = W.sum(axis=2).mean(axis=1)
    return out


def gamma_logpdf_torch(x, shape, scale):
    x = x.double()
    k = float(shape)
    th = float(scale)
    return (k - 1.0) * torch.log(x.clamp(min=1e-12)) - x / th - k * math.log(th) - math.lgamma(k)


class MechanisticPriorFull(Distribution):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([2]), validate_args=False)
        self.lut_b = torch.tensor(lut_betas, dtype=torch.float64)
        self.lut_sr = torch.tensor(lut_seed_rowsums, dtype=torch.float64)
        ld_seed = log_dist[seed_idx].astype(np.float64).copy()
        for k, sk in enumerate(np.asarray(seed_idx).tolist()):
            ld_seed[k, sk] = np.inf
        self.log_dist_seed = torch.tensor(ld_seed, dtype=torch.float64)
        self.lut_lo = float(lut_betas[0])
        self.lut_hi = float(lut_betas[-1])

    def _sr_safe(self, beta):
        b = beta.double()
        in_range = (b >= self.lut_lo) & (b <= self.lut_hi)
        sr = torch.empty_like(b)
        if in_range.any():
            sr[in_range] = interp_torch(b[in_range], self.lut_b, self.lut_sr)
        if (~in_range).any():
            ld_seed_dev = self.log_dist_seed.to(b.device)
            sr[~in_range] = compute_sr_exact_torch(b[~in_range], ld_seed_dev)
        return sr

    def sample(self, sample_shape=torch.Size()):
        n = max(1, int(torch.Size(sample_shape).numel()))
        b = torch._standard_gamma(torch.full((n,), BETA_SHAPE, dtype=torch.float64)) * BETA_SCALE
        r0 = torch._standard_gamma(torch.full((n,), R0_SHAPE, dtype=torch.float64)) * R0_SCALE
        sr = self._sr_safe(b)
        a = r0 / (IP_FIXED * sr)
        t = torch.stack([a.float(), b.float()], dim=1)
        return t.reshape(sample_shape + (2,)) if sample_shape else t.squeeze(0)

    def log_prob(self, theta):
        a = theta[..., 0].double()
        b = theta[..., 1].double()
        sr = self._sr_safe(b)
        r0 = a * IP_FIXED * sr
        ok = (a > 0) & (b > 0) & (r0 > 0) & torch.isfinite(sr) & torch.isfinite(r0)
        lp_b = gamma_logpdf_torch(b, BETA_SHAPE, BETA_SCALE)
        lp_r0 = gamma_logpdf_torch(r0, R0_SHAPE, R0_SCALE)
        lp = lp_b + lp_r0 + math.log(IP_FIXED) + torch.log(sr.clamp(min=1e-12))
        return torch.where(ok, lp, torch.full_like(lp, -float("inf"))).float()


class MechanisticPriorPartial(Distribution):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([3]), validate_args=False)
        self._ab = MechanisticPriorFull(lut_betas, lut_seed_rowsums, log_dist, seed_idx)

    def sample(self, sample_shape=torch.Size()):
        n = max(1, int(torch.Size(sample_shape).numel()))
        ab = self._ab.sample((n,))
        rho = RHO_EPS + (1.0 - 2.0 * RHO_EPS) * torch.rand(n, 1, dtype=torch.float32)
        t = torch.cat([ab, rho], dim=1)
        return t.reshape(sample_shape + (3,)) if sample_shape else t.squeeze(0)

    def log_prob(self, theta):
        lp_ab = self._ab.log_prob(theta[..., :2])
        rho = theta[..., 2]
        ok_rho = (rho > 0.0) & (rho < 1.0)
        lp_rho = torch.where(ok_rho, torch.zeros_like(rho), torch.full_like(rho, -float("inf")))
        return lp_ab + lp_rho


class MechanisticPriorGamma(Distribution):
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([3]), validate_args=False)
        self.lut_b = torch.tensor(lut_betas, dtype=torch.float64)
        self.lut_sr = torch.tensor(lut_seed_rowsums, dtype=torch.float64)
        ld_seed = log_dist[seed_idx].astype(np.float64).copy()
        for k, sk in enumerate(np.asarray(seed_idx).tolist()):
            ld_seed[k, sk] = np.inf
        self.log_dist_seed = torch.tensor(ld_seed, dtype=torch.float64)
        self.lut_lo = float(lut_betas[0])
        self.lut_hi = float(lut_betas[-1])

    def _sr_safe(self, beta):
        b = beta.double()
        in_range = (b >= self.lut_lo) & (b <= self.lut_hi)
        sr = torch.empty_like(b)
        if in_range.any():
            sr[in_range] = interp_torch(b[in_range], self.lut_b, self.lut_sr)
        if (~in_range).any():
            ld_seed_dev = self.log_dist_seed.to(b.device)
            sr[~in_range] = compute_sr_exact_torch(b[~in_range], ld_seed_dev)
        return sr

    def sample(self, sample_shape=torch.Size()):
        n = max(1, int(torch.Size(sample_shape).numel()))
        b = torch._standard_gamma(torch.full((n,), BETA_SHAPE, dtype=torch.float64)) * BETA_SCALE
        r0 = torch._standard_gamma(torch.full((n,), R0_SHAPE, dtype=torch.float64)) * R0_SCALE
        d = torch.exp(MU_D + SIGMA_D * torch.randn(n, dtype=torch.float64)).clamp_min(D_MIN)
        g = -torch.log1p(-1.0 / d)
        sr = self._sr_safe(b)
        a = r0 / (d * sr)
        t = torch.stack([a.float(), b.float(), g.float()], dim=1)
        return t.reshape(sample_shape + (3,)) if sample_shape else t.squeeze(0)

    def log_prob(self, theta):
        a = theta[..., 0].double()
        b = theta[..., 1].double()
        g = theta[..., 2].double()
        ok0 = (a > 0) & (b > 0) & (g > 0)
        d = 1.0 / (1.0 - torch.exp(-g)).clamp(min=1e-12)
        sr = self._sr_safe(b)
        r0 = a * d * sr
        ok = ok0 & (d > D_MIN) & (r0 > 0) & torch.isfinite(d) & torch.isfinite(sr) & torch.isfinite(r0)
        ld = torch.log(d.clamp(min=1e-12))
        lpd = (
            -0.5 * math.log(2 * math.pi)
            - math.log(SIGMA_D)
            - 0.5 * ((ld - MU_D) / SIGMA_D) ** 2
        )
        lp_b = gamma_logpdf_torch(b, BETA_SHAPE, BETA_SCALE)
        lp_r0 = gamma_logpdf_torch(r0, R0_SHAPE, R0_SCALE)
        lp = (
            lp_b
            + lp_r0
            + lpd
            + 2.0 * ld
            + torch.log(sr.clamp(min=1e-12))
            + torch.log((d - 1.0).clamp(min=1e-12))
        )
        return torch.where(ok, lp, torch.full_like(lp, -float("inf"))).float()


def make_prior(scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    if scenario == "full":
        return MechanisticPriorFull(lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    if scenario == "partial":
        return MechanisticPriorPartial(lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    return MechanisticPriorGamma(lut_betas, lut_seed_rowsums, log_dist, seed_idx)


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


def inf_time_to_daily(x_batch, T_len):
    n = x_batch.shape[0]
    curve = np.zeros((n, T_len), dtype=np.float32)
    for t in range(1, T_len + 1):
        curve[:, t - 1] = (x_batch == t).sum(axis=1)
    return curve


def load_cnn(split_dir):
    tr = np.load(split_dir / "train.npz", allow_pickle=True)
    te = np.load(split_dir / "test.npz", allow_pickle=True)
    T_len = int(tr["T"])
    th_tr = tr["theta"].astype(np.float32)
    th_te = te["theta"].astype(np.float32)
    c_tr = np.log1p(inf_time_to_daily(tr["x"].astype(np.int16), T_len))
    c_te = np.log1p(inf_time_to_daily(te["x"].astype(np.int16), T_len))
    mean = c_tr.mean(axis=0, keepdims=True)
    std = np.where(c_tr.std(axis=0, keepdims=True) < 1e-6, 1.0, c_tr.std(axis=0, keepdims=True))
    x_tr = ((c_tr - mean) / std)[:, None, :].astype(np.float32)
    x_te = ((c_te - mean) / std)[:, None, :].astype(np.float32)
    return th_tr, x_tr, th_te, x_te, T_len


def load_gnn(split_dir):
    tr = np.load(split_dir / "train.npz", allow_pickle=True)
    te = np.load(split_dir / "test.npz", allow_pickle=True)
    T_len = int(tr["T"])
    th_tr = tr["theta"].astype(np.float32)
    th_te = te["theta"].astype(np.float32)
    s = -1.0 / T_len
    enc = lambda x: np.where(x >= 0, x.astype(np.float32) / T_len, s).astype(np.float32)
    return th_tr, enc(tr["x"].astype(np.int16)), th_te, enc(te["x"].astype(np.int16)), T_len


class CNN1D(nn.Module):
    def __init__(self, T_len, emb_dim=64, channels1=32, channels2=128, pool_out=8, dropout=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels1, kernel_size=5, padding=2),
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


class SAGELayer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.ls = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.Linear(d_in, d_out, bias=False)
        self.b = nn.Parameter(torch.zeros(d_out))
        self.nm = nn.LayerNorm(d_out)
        self.dr = nn.Dropout(dropout)

    def forward(self, h, idx):
        return self.dr(F.relu(self.nm(self.ls(h) + self.ln(h[:, idx, :].mean(2)) + self.b)))


class GNN(nn.Module):
    def __init__(self, positions, region_side, T_len, emb_dim=32, hidden=64, n_layers=3, k=5, dropout=0.05):
        super().__init__()
        self.T = float(T_len)
        dists = squareform(pdist(positions, "euclidean"))
        np.fill_diagonal(dists, np.inf)
        knn = np.argsort(dists, axis=1)[:, :k].astype(np.int64)
        self.register_buffer("knn", torch.tensor(knn, dtype=torch.long))
        self.register_buffer("pos_n", torch.tensor(positions / region_side, dtype=torch.float32))
        self.enc = nn.Linear(4, hidden)
        self.lyrs = nn.ModuleList([SAGELayer(hidden, hidden, dropout) for _ in range(n_layers)])
        self.dr = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        s = -1.0 / self.T
        obs = (x > s + 1e-6).float().unsqueeze(-1)
        day = x.clamp(min=0).unsqueeze(-1)
        pos = self.pos_n.unsqueeze(0).expand(B, -1, -1)
        h = F.relu(self.enc(torch.cat([obs, day, pos], -1)))
        for l in self.lyrs:
            h = l(h, self.knn)
        return self.head(h.mean(1))


class WeightedSAGELayer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.ls = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.Linear(d_in, d_out, bias=False)
        self.b = nn.Parameter(torch.zeros(d_out))
        self.lam_param = nn.Parameter(torch.tensor(0.0))
        self.nm = nn.LayerNorm(d_out)
        self.dr = nn.Dropout(dropout)

    def lam(self):
        return self.lam_param

    def forward(self, h, idx, log_d):
        w = torch.softmax(-self.lam() * log_d, dim=1)
        agg = (h[:, idx, :] * w[None, :, :, None]).sum(2)
        return self.dr(F.relu(self.nm(self.ls(h) + self.ln(agg) + self.b)))


class WeightedGNN(nn.Module):
    def __init__(self, positions, region_side, T_len, emb_dim=32, hidden=64, n_layers=3, k=5, dropout=0.05):
        super().__init__()
        self.T = float(T_len)
        dists = squareform(pdist(positions, "euclidean"))
        np.fill_diagonal(dists, np.inf)
        knn = np.argsort(dists, axis=1)[:, :k].astype(np.int64)
        knn_d = np.maximum(np.take_along_axis(dists, knn, axis=1), 1e-6)
        self.register_buffer("knn", torch.tensor(knn, dtype=torch.long))
        self.register_buffer("knn_logd", torch.tensor(np.log(knn_d / np.median(knn_d)), dtype=torch.float32))
        self.register_buffer("pos_n", torch.tensor(positions / region_side, dtype=torch.float32))
        self.enc = nn.Linear(4, hidden)
        self.lyrs = nn.ModuleList([WeightedSAGELayer(hidden, hidden, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        s = -1.0 / self.T
        obs = (x > s + 1e-6).float().unsqueeze(-1)
        day = x.clamp(min=0).unsqueeze(-1)
        pos = self.pos_n.unsqueeze(0).expand(B, -1, -1)
        h = F.relu(self.enc(torch.cat([obs, day, pos], -1)))
        for l in self.lyrs:
            h = l(h, self.knn, self.knn_logd)
        return self.head(h.mean(1))

    def lambdas(self):
        return [float(l.lam().detach().cpu()) for l in self.lyrs]


def _sr_np_safe(b_arr, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    b = np.asarray(b_arr, dtype=np.float64)
    sr = np.empty_like(b)
    in_range = (b >= lut_betas[0]) & (b <= lut_betas[-1])
    sr[in_range] = np.interp(b[in_range], lut_betas, lut_seed_rowsums)
    if (~in_range).any():
        sr[~in_range] = compute_sr_exact_np(b[~in_range], log_dist, seed_idx)
    return sr


def valid_mask_full(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    a = samples[:, 0].astype(np.float64)
    b = samples[:, 1].astype(np.float64)
    positive = (a > 0) & (b > 0)
    sr = np.full_like(b, np.nan, dtype=np.float64)
    if positive.any():
        sr[positive] = _sr_np_safe(b[positive], lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    r0 = a * IP_FIXED * sr
    return positive & np.isfinite(sr) & np.isfinite(r0) & (sr > 0) & (r0 > 0)


def valid_mask_partial(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    base = valid_mask_full(samples[:, :2], lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    rho = samples[:, 2].astype(np.float64)
    return base & (rho > 0.0) & (rho < 1.0)


def valid_mask_gamma(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    a = samples[:, 0].astype(np.float64)
    b = samples[:, 1].astype(np.float64)
    g = samples[:, 2].astype(np.float64)
    positive = (a > 0) & (b > 0) & (g > 0)
    p_ir = 1.0 - np.exp(-g)
    ok1 = positive & (p_ir > 0) & (p_ir < 1)
    d = np.full_like(g, np.nan, dtype=np.float64)
    d[ok1] = 1.0 / p_ir[ok1]
    sr = np.full_like(b, np.nan, dtype=np.float64)
    if ok1.any():
        sr[ok1] = _sr_np_safe(b[ok1], lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    r0 = a * d * sr
    return ok1 & np.isfinite(d) & np.isfinite(sr) & np.isfinite(r0) & (d > D_MIN) & (sr > 0) & (r0 > 0)


def valid_mask_by_scenario(samples, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    if scenario == "full":
        return valid_mask_full(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    if scenario == "partial":
        return valid_mask_partial(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    return valid_mask_gamma(samples, lut_betas, lut_seed_rowsums, log_dist, seed_idx)


def sample_valid_posterior(posterior, x_obs, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx, n_post):
    batch_draws = max(2000, 4 * n_post)
    kept = []
    tries = 0
    max_tries = 40
    while sum(arr.shape[0] for arr in kept) < n_post and tries < max_tries:
        tries += 1
        raw = posterior.sample((batch_draws,), x=x_obs, show_progress_bars=False).detach().cpu().numpy()
        mask = valid_mask_by_scenario(raw, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
        good = raw[mask]
        if good.shape[0] > 0:
            kept.append(good.astype(np.float32))
    n_kept = sum(arr.shape[0] for arr in kept)
    if n_kept < n_post:
        return None
    return np.concatenate(kept, axis=0)[:n_post]


def train_eval(th_tr, x_tr, th_te, x_te, prior, emb, hps, device, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    set_seed(hps.get("seed", SEED))
    de_fn = posterior_nn(
        model=hps["flow_model"],
        embedding_net=emb,
        hidden_features=hps["hidden_features"],
        num_transforms=hps["num_transforms"],
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
    tr_sec = time.perf_counter() - t0
    val_lp = float("nan")
    n_epochs = 0
    best_epoch = -1
    try:
        vl = inf.summary.get("validation_log_probs", [])
        if vl:
            val_lp = float(np.max(vl))
            n_epochs = len(vl)
            best_epoch = int(np.argmax(vl))
    except Exception:
        pass

    post = inf.build_posterior(de)
    n, p = th_te.shape
    samp = np.empty((n, N_EVAL_POST, p), dtype=np.float32)
    valid_obs = 0
    for i in range(n):
        xo = torch.tensor(x_te[i:i + 1], dtype=torch.float32, device=device)
        draw = sample_valid_posterior(post, xo, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx, N_EVAL_POST)
        if draw is None:
            samp[i] = np.nan
        else:
            samp[i] = draw
            valid_obs += 1

    if valid_obs == 0:
        mae = float("inf")
    else:
        med = np.nanmedian(samp, axis=1)
        mae = float(np.nanmean(np.abs(med - th_te)))

    return mae, val_lp, tr_sec, valid_obs, n_epochs, best_epoch


def cnn_objective(th_tr, x_tr, th_te, x_te, prior, T_len, device, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    def obj(trial):
        lr = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        nt = trial.suggest_int("num_transforms", 3, 10)
        ed = trial.suggest_categorical("emb_dim", [32, 64, 128, 256])
        c1 = trial.suggest_categorical("channels1", [16, 32, 64])
        c2 = trial.suggest_categorical("channels2", [64, 128, 256])
        hf = trial.suggest_categorical("hidden_features", [32, 64, 128])
        hps = {
            **FIXED_HPARAMS,
            "learning_rate": lr,
            "num_transforms": nt,
            "emb_dim": ed,
            "channels1": c1,
            "channels2": c2,
            "hidden_features": hf,
            "seed": SEED + trial.number,
        }
        emb = CNN1D(
            T_len,
            emb_dim=ed,
            channels1=c1,
            channels2=c2,
            pool_out=hps["pool_out"],
            dropout=hps["dropout"],
        ).to(device)
        mae, vl, ts, valid_obs, n_ep, best_ep = train_eval(
            th_tr,
            x_tr,
            th_te,
            x_te,
            prior,
            emb,
            hps,
            device,
            scenario,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
        )
        trial.set_user_attr("val_log_prob", vl)
        trial.set_user_attr("train_sec", ts)
        trial.set_user_attr("valid_obs", valid_obs)
        trial.set_user_attr("n_epochs", n_ep)
        trial.set_user_attr("best_epoch", best_ep)
        print(
            f"  CNN {trial.number:3d}  lr={lr:.1e} nt={nt} ed={ed} "
            f"c1={c1} c2={c2} hf={hf}  mae={mae:.4f}  valid={valid_obs}/{len(th_te)}  "
            f"epochs={n_ep}(best={best_ep})  t={ts:.0f}s"
        )
        return mae

    return obj


def gnn_objective(th_tr, x_tr, th_te, x_te, prior, positions, region_side, T_len, device, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx):
    def obj(trial):
        lr = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        nt = trial.suggest_int("num_transforms", 3, 10)
        ed = trial.suggest_categorical("emb_dim", [16, 32, 64, 128])
        gh = trial.suggest_categorical("gnn_hidden", [32, 64, 128])
        gl = trial.suggest_int("gnn_layers", 2, 5)
        gk = trial.suggest_categorical("gnn_k", [5, 10, 15, 20])
        hf = trial.suggest_categorical("hidden_features", [32, 64, 128])
        hps = {
            **FIXED_HPARAMS,
            "learning_rate": lr,
            "num_transforms": nt,
            "emb_dim": ed,
            "gnn_hidden": gh,
            "gnn_layers": gl,
            "gnn_k": gk,
            "hidden_features": hf,
            "seed": SEED + trial.number,
        }
        emb = GNN(
            positions,
            region_side,
            T_len,
            emb_dim=ed,
            hidden=gh,
            n_layers=gl,
            k=gk,
            dropout=hps["dropout"],
        ).to(device)
        mae, vl, ts, valid_obs, n_ep, best_ep = train_eval(
            th_tr,
            x_tr,
            th_te,
            x_te,
            prior,
            emb,
            hps,
            device,
            scenario,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
        )
        trial.set_user_attr("val_log_prob", vl)
        trial.set_user_attr("train_sec", ts)
        trial.set_user_attr("valid_obs", valid_obs)
        trial.set_user_attr("n_epochs", n_ep)
        trial.set_user_attr("best_epoch", best_ep)
        print(
            f"  GNN {trial.number:3d}  lr={lr:.1e} nt={nt} ed={ed} "
            f"gh={gh} gl={gl} gk={gk} hf={hf}  mae={mae:.4f}  valid={valid_obs}/{len(th_te)}  "
            f"epochs={n_ep}(best={best_ep})  t={ts:.0f}s"
        )
        return mae

    return obj


def run_study(objective, n_trials):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    b = study.best_trial
    return {
        "best_config": b.params,
        "best_mae": float(b.value),
        "best_val_lp": float(b.user_attrs.get("val_log_prob", float("nan"))),
        "best_valid_obs": int(b.user_attrs.get("valid_obs", 0)),
        "best_n_epochs": int(b.user_attrs.get("n_epochs", 0)),
        "best_epoch": int(b.user_attrs.get("best_epoch", -1)),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "mae": float(t.value) if t.value is not None else None,
                "val_lp": float(t.user_attrs.get("val_log_prob", float("nan"))),
                "valid_obs": int(t.user_attrs.get("valid_obs", 0)),
                "n_epochs": int(t.user_attrs.get("n_epochs", 0)),
                "best_epoch": int(t.user_attrs.get("best_epoch", -1)),
            }
            for t in study.trials
            if t.value is not None
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "partial", "gamma"])
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    scenario = args.scenario
    cfg = load_config()
    lut_betas = cfg["lut_betas"]
    lut_seed_rowsums = cfg["lut_seed_rowsums"]
    log_dist = cfg["log_dist"]
    seed_idx = cfg["seed_idx"]
    prior = make_prior(scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
    pilot = STRICT_ROOT / scenario / "pilot"
    out_dir = STRICT_ROOT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scenario={scenario}  device={DEVICE}  n_trials={args.n_trials}")
    print(f"prior: beta~Gamma({BETA_SHAPE}, {BETA_SCALE:.4g}), R0~Gamma({R0_SHAPE}, {R0_SCALE:.4g})")
    print(f"LUT range: [{lut_betas[0]:.2f}, {lut_betas[-1]:.2f}], exact fallback outside LUT")
    print(f"max_epochs={FIXED_HPARAMS['max_num_epochs']}, early_stop_patience={FIXED_HPARAMS['stop_after_epochs']}")

    print("\n=== CNN tuning ===")
    th_tr, x_cnn_tr, th_te, x_cnn_te, T_len = load_cnn(pilot)
    t0 = time.perf_counter()
    cnn_res = run_study(
        cnn_objective(
            th_tr,
            x_cnn_tr,
            th_te,
            x_cnn_te,
            prior,
            T_len,
            DEVICE,
            scenario,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
        ),
        args.n_trials,
    )
    cnn_wall = (time.perf_counter() - t0) / 60
    print(f"CNN best mae={cnn_res['best_mae']:.4f}  {cnn_res['best_config']}")

    print("\n=== GNN tuning ===")
    th_tr2, x_gnn_tr, th_te2, x_gnn_te, _ = load_gnn(pilot)
    t0 = time.perf_counter()
    gnn_res = run_study(
        gnn_objective(
            th_tr2,
            x_gnn_tr,
            th_te2,
            x_gnn_te,
            prior,
            cfg["positions"],
            cfg["region_side"],
            T_len,
            DEVICE,
            scenario,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
        ),
        args.n_trials,
    )
    gnn_wall = (time.perf_counter() - t0) / 60
    print(f"GNN best mae={gnn_res['best_mae']:.4f}  {gnn_res['best_config']}")

    out = {
        "scenario": scenario,
        "prior": {
            "beta": {"family": "Gamma", "shape": BETA_SHAPE, "scale": BETA_SCALE},
            "R0": {"family": "Gamma", "shape": R0_SHAPE, "scale": R0_SCALE},
        },
        "training_settings": {
            "max_num_epochs": FIXED_HPARAMS["max_num_epochs"],
            "stop_after_epochs": FIXED_HPARAMS["stop_after_epochs"],
            "batch_size": FIXED_HPARAMS["batch_size"],
        },
        "cnn": {**cnn_res, "tuning_wall_min": float(cnn_wall)},
        "gnn": {**gnn_res, "tuning_wall_min": float(gnn_wall)},
    }

    if scenario == "partial":
        out["prior"]["rho"] = {"family": "Uniform", "low": 0.0, "high": 1.0}

    if scenario == "gamma":
        out["prior"]["d"] = {"family": "LogNormal", "mu": MU_D, "sigma": SIGMA_D, "lower": D_MIN}
        out["prior"]["gamma"] = {"family": "induced", "from": "-log(1 - 1 / d)"}

    with open(out_dir / "best_hparams.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {out_dir / 'best_hparams.json'}")


if __name__ == "__main__":
    main()