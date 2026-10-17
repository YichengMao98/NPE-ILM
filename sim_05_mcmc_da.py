import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import gamma as gamma_dist

from sim_09_mcmc_diagnostics import rhat_rank, ess_bulk, ess_tail

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
CONFIG_ROOT = Path(os.environ.get("ILM_CONFIG_ROOT", ARC_ROOT / "configs")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()

N_CPUS = max(1, int(os.environ.get("ILM_N_CPUS", os.cpu_count() or 1)))

T = 40
IP_FIXED = 3

BETA_SHAPE = 6.0
BETA_SCALE = 1.0 / 4.0
R0_SHAPE = 10.0
R0_SCALE = 1.0 / 4.0
MU_D = math.log(3) - 0.27**2 / 2
SIGMA_D = 0.27
D_MIN = 1.001
RHO_EPS = 1e-6

SEED = 1234
N_OBS = 100

SCENARIOS = ("full", "gamma", "partial")
PARAM_NAMES = {
    "full": ("alpha", "beta"),
    "partial": ("alpha", "beta", "rho"),
    "gamma": ("alpha", "beta", "gamma"),
}
LATENT_NAMES = {
    "full": None,
    "gamma": "mean_infectious_steps",
    "partial": "n_hidden_infections",
}
DEFAULT_INNER = {"full": 0, "gamma": 50, "partial": 200}
OUT_DIRNAME = "mcmc"
LATENT_INIT = {
    "full": "none",
    "gamma": "durations drawn from the duration model given the initial gamma of each chain",
    "partial": "overdispersed: chain c starts with a fraction drawn from U(c/C, (c+1)/C) of the unobserved individuals infected",
}

N_CHAINS = 4
N_ITER = 100_000
N_BURNIN = 20_000
STORE_THIN = 10
COMPARE_DRAWS_PER_CHAIN = 750

ADAPT_START = 100
GELMAN_SCALE = 2.38
MIXTURE_EPS = 0.05
SIGMA0_DIAG = 0.05
REBUILD_EVERY = 50
TARGET_ACC_1D = 0.44
LOG_TINY = math.log(1e-300)


def compute_sr_exact_np(betas, log_dist, seed_idx):
    betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    ld_seed = log_dist[seed_idx].astype(np.float64).copy()
    for k, sk in enumerate(seed_idx):
        ld_seed[k, sk] = np.inf
    out = np.empty(len(betas), dtype=np.float64)
    for s in range(0, len(betas), 500):
        e = min(s + 500, len(betas))
        W = np.exp(-betas[s:e, None, None] * ld_seed[None, :, :])
        out[s:e] = W.sum(axis=2).mean(axis=1)
    return out


def sr_safe(beta, lut_b, lut_sr, log_dist, seed_idx):
    if beta < lut_b[0] or beta > lut_b[-1]:
        return float(compute_sr_exact_np(np.array([beta]), log_dist, seed_idx)[0])
    return float(np.interp(beta, lut_b, lut_sr))


def log_prior_full(alpha, beta, lut_b, lut_sr, log_dist, seed_idx):
    if alpha <= 0 or beta <= 0:
        return -math.inf
    sr = sr_safe(beta, lut_b, lut_sr, log_dist, seed_idx)
    if sr <= 0:
        return -math.inf
    r0 = alpha * IP_FIXED * sr
    if r0 <= 0:
        return -math.inf
    lp_b = float(gamma_dist.logpdf(beta, a=BETA_SHAPE, scale=BETA_SCALE))
    lp_r0 = float(gamma_dist.logpdf(r0, a=R0_SHAPE, scale=R0_SCALE))
    return lp_b + lp_r0 + math.log(IP_FIXED) + math.log(sr)


def get_obs_idx(n_test, n_obs=N_OBS):
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_test, size=n_obs, replace=False))


def log1mexp(x):
    return np.log(np.maximum(-np.expm1(-np.asarray(x, dtype=np.float64)), 1e-300))


def log1mexp_scalar(x):
    return math.log(max(-math.expm1(-float(x)), 1e-300))


def log_prior_partial(alpha, beta, rho, lut_b, lut_sr, log_dist, seed_idx):
    if not (0.0 < rho < 1.0):
        return -math.inf
    return log_prior_full(alpha, beta, lut_b, lut_sr, log_dist, seed_idx)


def log_prior_gamma(alpha, beta, gamma, lut_b, lut_sr, log_dist, seed_idx):
    if alpha <= 0 or beta <= 0 or gamma <= 0:
        return -math.inf
    p_ir = 1.0 - math.exp(-gamma)
    if p_ir <= 0 or p_ir >= 1.0:
        return -math.inf
    d = 1.0 / p_ir
    if d <= D_MIN:
        return -math.inf
    sr = sr_safe(beta, lut_b, lut_sr, log_dist, seed_idx)
    if sr <= 0:
        return -math.inf
    r0 = alpha * d * sr
    if r0 <= 0:
        return -math.inf
    log_d = math.log(d)
    lp_d = -0.5 * math.log(2 * math.pi) - math.log(SIGMA_D) - 0.5 * ((log_d - MU_D) / SIGMA_D) ** 2
    lp_b = float(gamma_dist.logpdf(beta, a=BETA_SHAPE, scale=BETA_SCALE))
    lp_r0 = float(gamma_dist.logpdf(r0, a=R0_SHAPE, scale=R0_SCALE))
    return lp_b + lp_r0 + lp_d + log_d + math.log(sr) + math.log(d - 1.0)


def sample_prior_alpha_beta(rng, infectious_scale, lut_b, lut_sr, log_dist, seed_idx):
    beta = float(rng.gamma(BETA_SHAPE, BETA_SCALE))
    r0 = float(rng.gamma(R0_SHAPE, R0_SCALE))
    alpha = r0 / (infectious_scale * sr_safe(beta, lut_b, lut_sr, log_dist, seed_idx))
    return alpha, beta


def sample_prior_partial(rng, lut_b, lut_sr, log_dist, seed_idx):
    alpha, beta = sample_prior_alpha_beta(rng, IP_FIXED, lut_b, lut_sr, log_dist, seed_idx)
    return alpha, beta, float(rng.uniform(RHO_EPS, 1.0 - RHO_EPS))


def sample_prior_gamma(rng, lut_b, lut_sr, log_dist, seed_idx):
    beta = float(rng.gamma(BETA_SHAPE, BETA_SCALE))
    r0 = float(rng.gamma(R0_SHAPE, R0_SCALE))
    d = max(math.exp(rng.normal(MU_D, SIGMA_D)), D_MIN)
    gamma = -math.log(1.0 - 1.0 / d)
    alpha = r0 / (d * sr_safe(beta, lut_b, lut_sr, log_dist, seed_idx))
    return alpha, beta, gamma


class ILMState:
    def __init__(self, log_dist, z, dur, alpha, beta):
        self.ld = log_dist
        self.N = log_dist.shape[0]
        self.tt = np.arange(T)
        self.z = np.asarray(z, dtype=np.int64).copy()
        self.dur = np.asarray(dur, dtype=np.int64).copy()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.W = self.kernel(self.beta)
        self.rebuild()

    def kernel(self, beta):
        W = np.exp(-beta * self.ld)
        np.fill_diagonal(W, 0.0)
        return W

    def infectious_matrix(self, z, dur):
        t = self.tt[:, None]
        return ((z[None, :] >= 0) & (z[None, :] <= t) & (t < z[None, :] + dur[None, :])).astype(np.float64)

    def rebuild(self):
        t = self.tt[:, None]
        self.I = self.infectious_matrix(self.z, self.dur)
        self.S = self.I @ self.W
        self.SU = (self.z[None, :] < 0) | (self.z[None, :] > t + 1)
        self.NI = self.z[None, :] == t + 1
        self.rows_all()

    def rows_all(self):
        self.rowB = np.where(self.SU, self.S, 0.0).sum(axis=1)
        self.rowE = np.zeros(T)
        tt, jj = np.nonzero(self.NI)
        if tt.size:
            np.add.at(self.rowE, tt, log1mexp(self.alpha * self.S[tt, jj]))

    def rows_update(self, rows):
        if not rows:
            return
        R = np.fromiter(rows, dtype=np.int64, count=len(rows))
        S_R = self.S[R]
        self.rowB[R] = np.where(self.SU[R], S_R, 0.0).sum(axis=1)
        self.rowE[R] = 0.0
        tt, jj = np.nonzero(self.NI[R])
        if tt.size:
            np.add.at(self.rowE, R[tt], log1mexp(self.alpha * S_R[tt, jj]))

    def loglik(self):
        return float(self.rowE.sum() - self.alpha * self.rowB.sum())

    def loglik_params(self, alpha, beta):
        if beta != self.beta:
            W = self.kernel(beta)
            S = self.I @ W
        else:
            W, S = self.W, self.S
        B = np.where(self.SU, S, 0.0).sum()
        tt, jj = np.nonzero(self.NI)
        E = float(log1mexp(alpha * S[tt, jj]).sum()) if tt.size else 0.0
        return E - alpha * B, (W, S)

    def set_params(self, alpha, beta, cache):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.W, self.S = cache
        self.rows_all()

    def loglik_latent(self, z, dur):
        I = self.infectious_matrix(z, dur)
        S = I @ self.W
        t = self.tt[:, None]
        SU = (z[None, :] < 0) | (z[None, :] > t + 1)
        NI = z[None, :] == t + 1
        B = np.where(SU, S, 0.0).sum()
        tt, jj = np.nonzero(NI)
        E = float(log1mexp(self.alpha * S[tt, jj]).sum()) if tt.size else 0.0
        return E - self.alpha * B, (z.copy(), dur.copy(), I, S, SU, NI)

    def set_latent(self, cache):
        self.z, self.dur, self.I, self.S, self.SU, self.NI = cache
        self.rows_all()

    def rows_of(self, z_i, d_i):
        if z_i < 0:
            return set()
        return set(range(int(z_i), int(min(z_i + d_i, T))))

    def change(self, i, z_new, d_new):
        z_old = int(self.z[i])
        old_rows = self.rows_of(z_old, self.dur[i])
        new_rows = self.rows_of(z_new, d_new)
        w = self.W[i]
        for t in old_rows - new_rows:
            self.S[t] -= w
            self.I[t, i] = 0.0
        for t in new_rows - old_rows:
            self.S[t] += w
            self.I[t, i] = 1.0
        self.z[i] = z_new
        self.dur[i] = d_new
        changed = old_rows ^ new_rows
        if z_new != z_old:
            col = self.S[:, i]
            self.rowB -= np.where(self.SU[:, i], col, 0.0)
            if z_old >= 1:
                self.rowE[z_old - 1] -= log1mexp_scalar(self.alpha * col[z_old - 1])
            t = self.tt
            self.SU[:, i] = (z_new < 0) | (z_new > t + 1)
            self.NI[:, i] = z_new == t + 1
            self.rowB += np.where(self.SU[:, i], col, 0.0)
            if z_new >= 1:
                self.rowE[z_new - 1] += log1mexp_scalar(self.alpha * col[z_new - 1])
        self.rows_update(changed)


class AdaptiveRW:
    def __init__(self, dim, adapt_until):
        self.d = dim
        self.adapt_until = adapt_until
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))
        self.chol = None
        self.chol_n = -1

    def update(self, x, it):
        if it >= self.adapt_until:
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += np.outer(delta, x - self.mean)

    def propose(self, x, rng):
        if self.n < ADAPT_START:
            return x + rng.normal(0.0, SIGMA0_DIAG, size=self.d)
        if self.chol is None or (self.n != self.chol_n and (self.n % 50 == 0 or self.n < self.adapt_until)):
            cov = self.M2 / (self.n - 1) + 1e-7 * np.eye(self.d)
            self.chol = np.linalg.cholesky((GELMAN_SCALE ** 2 / self.d) * cov)
            self.chol_n = self.n
        if rng.random() < MIXTURE_EPS:
            return x + rng.normal(0.0, SIGMA0_DIAG, size=self.d)
        return x + self.chol @ rng.normal(size=self.d)


class ScaleRW:
    def __init__(self, log_scale, adapt_until):
        self.log_scale = log_scale
        self.adapt_until = adapt_until

    def update(self, accepted, it):
        if it < self.adapt_until:
            self.log_scale += (float(accepted) - TARGET_ACC_1D) / math.sqrt(it + 1.0)

    @property
    def scale(self):
        return math.exp(self.log_scale)


class Counter:
    def __init__(self, names):
        self.prop = {k: 0 for k in names}
        self.acc = {k: 0 for k in names}

    def add(self, name, accepted, record):
        if record:
            self.prop[name] += 1
            self.acc[name] += int(accepted)

    def rates(self):
        return {k: (self.acc[k] / self.prop[k] if self.prop[k] else float("nan")) for k in self.prop}


def accept(rng, log_ratio):
    return math.log(rng.random() + 1e-300) < log_ratio


class GammaSampler:
    moves = ("alpha_beta", "gamma_centered", "gamma_noncentered", "dur_prior", "dur_local")

    def __init__(self, z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_site=DEFAULT_INNER["gamma"], init=None):
        self.ld = log_dist
        self.lut = (lut_b, lut_sr, log_dist, seed_idx)
        self.rng = rng
        self.n_site = n_site
        z = np.asarray(z_obs, dtype=np.int64)
        N = len(z)
        seed_mask = np.zeros(N, dtype=bool)
        seed_mask[seed_idx] = True
        self.m = np.where(seed_mask, 1, 0).astype(np.int64)
        self.K = np.where(z >= 0, T - z, 0).astype(np.int64)
        self.latent = np.where((z >= 0) & (self.K > self.m))[0]
        if init is None:
            alpha, beta, gamma = sample_prior_gamma(rng, *self.lut)
            dur = np.zeros(N, dtype=np.int64)
            dur[z >= 0] = self.K[z >= 0]
            dur[self.latent] = self.draw_dur(self.latent, gamma)
        else:
            alpha, beta, gamma, dur = init
        self.gamma = float(gamma)
        self.state = ILMState(log_dist, z, dur, alpha, beta)
        self.ll = self.state.loglik()
        self.lp = log_prior_gamma(alpha, beta, gamma, *self.lut)
        self.rw = AdaptiveRW(2, adapt_until)
        self.g_c = ScaleRW(math.log(0.2), adapt_until)
        self.g_nc = ScaleRW(math.log(0.1), adapt_until)
        self.counter = Counter(self.moves)

    def draw_dur(self, idx, gamma):
        e = self.rng.exponential(size=len(idx))
        return np.minimum(self.m[idx] + np.floor(e / gamma).astype(np.int64), self.K[idx])

    def lp_dur(self, dur, gamma):
        c = dur[self.latent]
        m = self.m[self.latent]
        K = self.K[self.latent]
        return float(-gamma * np.sum(c - m) + np.sum(c < K) * log1mexp_scalar(gamma))

    def lp_dur_one(self, i, c, gamma):
        return -gamma * (c - self.m[i]) + (c < self.K[i]) * log1mexp_scalar(gamma)

    def params(self):
        return np.array([self.state.alpha, self.state.beta, self.gamma])

    def logpost(self):
        return self.ll + self.lp + self.lp_dur(self.state.dur, self.gamma)

    def latent_summary(self):
        return float(self.state.dur[self.latent].mean()) if self.latent.size else 0.0

    def step(self, it, record):
        rng = self.rng
        st = self.state
        x = np.array([math.log(st.alpha), st.beta])
        prop = self.rw.propose(x, rng)
        a_p, b_p = math.exp(prop[0]), float(prop[1])
        lp_p = log_prior_gamma(a_p, b_p, self.gamma, *self.lut)
        ok = False
        if not math.isinf(lp_p):
            ll_p, cache = st.loglik_params(a_p, b_p)
            if accept(rng, (ll_p + lp_p) - (self.ll + self.lp) + prop[0] - x[0]):
                st.set_params(a_p, b_p, cache)
                self.ll, self.lp = ll_p, lp_p
                ok = True
        self.counter.add("alpha_beta", ok, record)
        self.rw.update(np.array([math.log(st.alpha), st.beta]), it)

        g_p = self.gamma * math.exp(self.g_c.scale * rng.normal())
        lp_p = log_prior_gamma(st.alpha, st.beta, g_p, *self.lut)
        ok = False
        if not math.isinf(lp_p):
            r = (lp_p + self.lp_dur(st.dur, g_p)) - (self.lp + self.lp_dur(st.dur, self.gamma)) + math.log(g_p / self.gamma)
            if accept(rng, r):
                self.gamma, self.lp = g_p, lp_p
                ok = True
        self.counter.add("gamma_centered", ok, record)
        self.g_c.update(ok, it)

        ok = False
        if self.latent.size:
            idx = self.latent
            c = st.dur[idx]
            lo = self.gamma * (c - self.m[idx])
            hi = np.where(c < self.K[idx], self.gamma * (c - self.m[idx] + 1), np.inf)
            u = rng.random(len(idx))
            width = np.where(np.isfinite(hi), -np.expm1(-(hi - np.where(np.isfinite(hi), lo, 0.0))), 1.0)
            e = np.where(np.isfinite(hi), lo - np.log1p(-u * width), lo - np.log(u))
            g_p = self.gamma * math.exp(self.g_nc.scale * rng.normal())
            lp_p = log_prior_gamma(st.alpha, st.beta, g_p, *self.lut)
            if not math.isinf(lp_p):
                dur_p = st.dur.copy()
                dur_p[idx] = np.minimum(self.m[idx] + np.floor(e / g_p).astype(np.int64), self.K[idx])
                ll_p, cache = st.loglik_latent(st.z, dur_p)
                if accept(rng, (ll_p + lp_p) - (self.ll + self.lp) + math.log(g_p / self.gamma)):
                    st.set_latent(cache)
                    self.gamma, self.ll, self.lp = g_p, ll_p, lp_p
                    ok = True
        self.counter.add("gamma_noncentered", ok, record)
        self.g_nc.update(ok, it)

        if self.latent.size:
            for _ in range(self.n_site):
                i = int(rng.choice(self.latent))
                c = int(st.dur[i])
                if rng.random() < 0.5:
                    c_p = int(min(self.m[i] + math.floor(rng.exponential() / self.gamma), self.K[i]))
                    extra = 0.0
                    name = "dur_prior"
                else:
                    c_p = c + int(rng.choice((-2, -1, 1, 2)))
                    name = "dur_local"
                    if c_p < self.m[i] or c_p > self.K[i]:
                        self.counter.add(name, False, record)
                        continue
                    extra = self.lp_dur_one(i, c_p, self.gamma) - self.lp_dur_one(i, c, self.gamma)
                if c_p == c:
                    self.counter.add(name, True, record)
                    continue
                st.change(i, int(st.z[i]), c_p)
                ll_p = st.loglik()
                if accept(rng, ll_p - self.ll + extra):
                    self.ll = ll_p
                    self.counter.add(name, True, record)
                else:
                    st.change(i, int(st.z[i]), c)
                    self.counter.add(name, False, record)

        if (it + 1) % REBUILD_EVERY == 0:
            st.rebuild()
            self.ll = st.loglik()


class PartialSampler:
    moves = ("alpha_beta", "add", "remove", "shift_uniform", "shift_local")

    def __init__(self, z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_aug=DEFAULT_INNER["partial"], init=None,
                 chain_id=0, n_chains=1):
        self.lut = (lut_b, lut_sr, log_dist, seed_idx)
        self.rng = rng
        self.n_aug = n_aug
        z_obs = np.asarray(z_obs, dtype=np.int64)
        N = len(z_obs)
        seed_mask = np.zeros(N, dtype=bool)
        seed_mask[seed_idx] = True
        self.n_obs_inf = int(((z_obs >= 0) & ~seed_mask).sum())
        self.U = np.where((z_obs < 0) & ~seed_mask)[0]
        if init is None:
            alpha, beta, rho = sample_prior_partial(rng, *self.lut)
            z = self.init_hidden(z_obs, seed_mask, chain_id, n_chains)
        else:
            alpha, beta, rho, z = init
            z = np.asarray(z, dtype=np.int64).copy()
        dur = np.where(z >= 0, IP_FIXED, 0)
        self.rho = float(rho)
        self.state = ILMState(log_dist, z, dur, alpha, beta)
        self.ll = self.state.loglik()
        self.lp = log_prior_full(alpha, beta, *self.lut)
        self.rw = AdaptiveRW(2, adapt_until)
        self.counter = Counter(self.moves)

    def init_hidden(self, z_obs, seed_mask, chain_id, n_chains):
        z = z_obs.copy()
        frac = float(self.rng.uniform(chain_id / n_chains, (chain_id + 1) / n_chains))
        pick = self.U[self.rng.random(len(self.U)) < frac]
        times = z_obs[(z_obs > 0) & ~seed_mask]
        if times.size:
            z[pick] = self.rng.choice(times, size=len(pick))
        else:
            z[pick] = self.rng.integers(1, T + 1, size=len(pick))
        return z

    def hidden(self):
        return self.U[self.state.z[self.U] >= 0]

    def free(self):
        return self.U[self.state.z[self.U] < 0]

    def params(self):
        return np.array([self.state.alpha, self.state.beta, self.rho])

    def logpost(self):
        n_h = len(self.hidden())
        return self.ll + self.lp + self.n_obs_inf * math.log(self.rho) + n_h * math.log1p(-self.rho)

    def latent_summary(self):
        return float(len(self.hidden()))

    def add_proposal(self, free):
        st = self.state
        h = -np.expm1(-st.alpha * np.maximum(st.S[:, free], 0.0))
        n = h.size
        tot = h.sum()
        if tot > 0:
            q = 0.5 / n + 0.5 * h / tot
        else:
            q = np.full(h.shape, 1.0 / n)
        return q

    def step(self, it, record):
        rng = self.rng
        st = self.state
        x = np.array([math.log(st.alpha), st.beta])
        prop = self.rw.propose(x, rng)
        a_p, b_p = math.exp(prop[0]), float(prop[1])
        lp_p = log_prior_full(a_p, b_p, *self.lut)
        ok = False
        if not math.isinf(lp_p):
            ll_p, cache = st.loglik_params(a_p, b_p)
            if accept(rng, (ll_p + lp_p) - (self.ll + self.lp) + prop[0] - x[0]):
                st.set_params(a_p, b_p, cache)
                self.ll, self.lp = ll_p, lp_p
                ok = True
        self.counter.add("alpha_beta", ok, record)
        self.rw.update(np.array([math.log(st.alpha), st.beta]), it)

        n_h = len(self.hidden())
        self.rho = float(rng.beta(1.0 + self.n_obs_inf, 1.0 + n_h))
        self.rho = min(max(self.rho, RHO_EPS), 1.0 - RHO_EPS)

        log1m_rho = math.log1p(-self.rho)
        for _ in range(self.n_aug):
            u = rng.random()
            if u < 1.0 / 3.0:
                free = self.free()
                if free.size == 0:
                    self.counter.add("add", False, record)
                    continue
                q = self.add_proposal(free)
                k = int(rng.choice(q.size, p=(q / q.sum()).ravel()))
                t_idx, j_idx = divmod(k, free.size)
                j = int(free[j_idx])
                t_new = t_idx + 1
                q_fwd = q.ravel()[k]
                n_h = len(self.hidden())
                st.change(j, t_new, IP_FIXED)
                ll_p = st.loglik()
                r = ll_p - self.ll + log1m_rho - math.log(n_h + 1) - math.log(q_fwd)
                if accept(rng, r):
                    self.ll = ll_p
                    self.counter.add("add", True, record)
                else:
                    st.change(j, -1, 0)
                    self.counter.add("add", False, record)
            elif u < 2.0 / 3.0:
                hid = self.hidden()
                if hid.size == 0:
                    self.counter.add("remove", False, record)
                    continue
                j = int(rng.choice(hid))
                t_old = int(st.z[j])
                n_h = hid.size
                st.change(j, -1, 0)
                ll_p = st.loglik()
                free = self.free()
                q = self.add_proposal(free)
                j_idx = int(np.nonzero(free == j)[0][0])
                q_rev = q[t_old - 1, j_idx]
                r = ll_p - self.ll - log1m_rho + math.log(q_rev) + math.log(n_h)
                if accept(rng, r):
                    self.ll = ll_p
                    self.counter.add("remove", True, record)
                else:
                    st.change(j, t_old, IP_FIXED)
                    self.counter.add("remove", False, record)
            else:
                hid = self.hidden()
                local = rng.random() < 0.5
                name = "shift_local" if local else "shift_uniform"
                if hid.size == 0:
                    self.counter.add(name, False, record)
                    continue
                j = int(rng.choice(hid))
                t_old = int(st.z[j])
                t_new = t_old + int(rng.choice((-2, -1, 1, 2))) if local else int(rng.integers(1, T + 1))
                if t_new < 1 or t_new > T:
                    self.counter.add(name, False, record)
                    continue
                if t_new == t_old:
                    self.counter.add(name, True, record)
                    continue
                st.change(j, t_new, IP_FIXED)
                ll_p = st.loglik()
                if accept(rng, ll_p - self.ll):
                    self.ll = ll_p
                    self.counter.add(name, True, record)
                else:
                    st.change(j, t_old, IP_FIXED)
                    self.counter.add(name, False, record)

        if (it + 1) % REBUILD_EVERY == 0:
            st.rebuild()
            self.ll = st.loglik()


class FullSampler:
    moves = ("alpha_beta",)

    def __init__(self, z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_inner=0, init=None):
        self.lut = (lut_b, lut_sr, log_dist, seed_idx)
        self.rng = rng
        z = np.asarray(z_obs, dtype=np.int64)
        if init is None:
            alpha, beta = sample_prior_alpha_beta(rng, IP_FIXED, *self.lut)
        else:
            alpha, beta = init
        self.state = ILMState(log_dist, z, np.where(z >= 0, IP_FIXED, 0), alpha, beta)
        self.ll = self.state.loglik()
        self.lp = log_prior_full(alpha, beta, *self.lut)
        self.rw = AdaptiveRW(2, adapt_until)
        self.counter = Counter(self.moves)

    def params(self):
        return np.array([self.state.alpha, self.state.beta])

    def logpost(self):
        return self.ll + self.lp

    def latent_summary(self):
        return float("nan")

    def step(self, it, record):
        rng = self.rng
        st = self.state
        x = np.array([math.log(st.alpha), st.beta])
        prop = self.rw.propose(x, rng)
        a_p, b_p = math.exp(prop[0]), float(prop[1])
        lp_p = log_prior_full(a_p, b_p, *self.lut)
        ok = False
        if not math.isinf(lp_p):
            ll_p, cache = st.loglik_params(a_p, b_p)
            if accept(rng, (ll_p + lp_p) - (self.ll + self.lp) + prop[0] - x[0]):
                st.set_params(a_p, b_p, cache)
                self.ll, self.lp = ll_p, lp_p
                ok = True
        self.counter.add("alpha_beta", ok, record)
        self.rw.update(np.array([math.log(st.alpha), st.beta]), it)


SAMPLERS = {"full": FullSampler, "gamma": GammaSampler, "partial": PartialSampler}


def build_sampler(scenario, z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_inner, chain_id=0, n_chains=1):
    if scenario == "gamma":
        return GammaSampler(z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_site=n_inner)
    if scenario == "partial":
        return PartialSampler(z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until, n_aug=n_inner,
                              chain_id=chain_id, n_chains=n_chains)
    return FullSampler(z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, adapt_until)


def run_chain(args):
    scenario, chain_id, n_chains, z_obs, seed_idx, log_dist, lut_b, lut_sr, n_iter, n_burnin, n_inner, seed = args
    rng = np.random.default_rng(seed)
    sampler = build_sampler(scenario, z_obs, seed_idx, log_dist, lut_b, lut_sr, rng, n_burnin, n_inner,
                            chain_id=chain_id, n_chains=n_chains)
    init_latent = sampler.latent_summary()
    n_keep = n_iter - n_burnin
    n_par = len(PARAM_NAMES[scenario])
    samples = np.empty((n_keep, n_par), dtype=np.float64)
    ll = np.empty(n_keep)
    lpost = np.empty(n_keep)
    latent = np.empty(n_keep)
    t0 = time.process_time()
    w0 = time.perf_counter()
    for it in range(n_iter):
        record = it >= n_burnin
        sampler.step(it, record)
        if record:
            k = it - n_burnin
            samples[k] = sampler.params()
            ll[k] = sampler.ll
            lpost[k] = sampler.logpost()
            latent[k] = sampler.latent_summary()
        if (it + 1) % 10000 == 0:
            print(f"    chain {chain_id} iter {it + 1}/{n_iter}  ll={sampler.ll:.1f}  "
                  f"params={np.round(sampler.params(), 4).tolist()}  {time.perf_counter() - w0:.0f}s", flush=True)
    return {
        "samples": samples,
        "loglik": ll,
        "logpost": lpost,
        "latent": latent,
        "acceptance": sampler.counter.rates(),
        "init_latent": float(init_latent),
        "cpu_sec": float(time.process_time() - t0),
    }


def chain_diagnostics(x):
    return {"rhat_rank": rhat_rank(x), "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x)}


def load_inputs(scenario):
    cfg = np.load(CONFIG_ROOT / "ilm_config.npz", allow_pickle=True)
    log_dist = cfg["log_dist"].astype(np.float64)
    lut_b = cfg["lut_betas"].astype(np.float64)
    lut_sr = cfg["lut_seed_rowsums"].astype(np.float64)
    seed_idx = cfg["seed_idx"].astype(np.int64)
    test = np.load(STRICT_ROOT / scenario / "final" / "test.npz", allow_pickle=True)
    stored = [str(p) for p in test["param_names"]]
    if stored != list(PARAM_NAMES[scenario]):
        raise ValueError(f"Parameter order mismatch: {stored}")
    return log_dist, lut_b, lut_sr, seed_idx, test["theta"].astype(np.float32), test["x"].astype(np.int64)


def run_one_obs(scenario, obs_i, n_obs, n_chains, n_iter, n_burnin, n_inner, store_thin, n_procs):
    log_dist, lut_b, lut_sr, seed_idx, theta_test, x_test = load_inputs(scenario)
    obs_idx = get_obs_idx(len(theta_test), n_obs)
    if obs_i < 0 or obs_i >= len(obs_idx):
        raise ValueError(f"obs_i={obs_i} out of range [0, {len(obs_idx) - 1}]")
    obs_id = int(obs_idx[obs_i])
    z_obs = x_test[obs_id]
    true_theta = theta_test[obs_id]
    param_names = PARAM_NAMES[scenario]
    latent_name = LATENT_NAMES[scenario]
    out_dir = STRICT_ROOT / scenario / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scenario={scenario}  obs_i={obs_i}  obs_id={obs_id}  chains={n_chains}  iter={n_iter}  "
          f"burnin={n_burnin}  inner={n_inner}  store_thin={store_thin}  procs={n_procs}", flush=True)
    print("  true: " + "  ".join(f"{p}={v:.4f}" for p, v in zip(param_names, true_theta))
          + f"  n_infected_observed={int((z_obs >= 0).sum())}", flush=True)

    rng_init = np.random.default_rng(SEED + obs_id * 100 + 7)
    args = [
        (scenario, c, n_chains, z_obs, seed_idx, log_dist, lut_b, lut_sr, n_iter, n_burnin, n_inner,
         int(rng_init.integers(1, 10**9)))
        for c in range(n_chains)
    ]
    w0 = time.perf_counter()
    with Pool(processes=max(1, min(n_chains, n_procs))) as pool:
        results = pool.map(run_chain, args)
    wall = time.perf_counter() - w0

    samples = np.stack([r["samples"] for r in results])
    loglik = np.stack([r["loglik"] for r in results])
    logpost = np.stack([r["logpost"] for r in results])
    latent = np.stack([r["latent"] for r in results])
    cpu = float(sum(r["cpu_sec"] for r in results))

    diag = {p: chain_diagnostics(samples[:, :, j]) for j, p in enumerate(param_names)}
    diag["loglik"] = chain_diagnostics(loglik)
    diag["logpost"] = chain_diagnostics(logpost)
    if latent_name is not None:
        diag[latent_name] = chain_diagnostics(latent)

    keep = slice(None, None, store_thin)
    arrays = dict(
        samples=samples[:, keep, :].astype(np.float32),
        loglik=loglik[:, keep].astype(np.float32),
        logpost=logpost[:, keep].astype(np.float32),
        theta=true_theta.astype(np.float32),
        obs_id=np.int32(obs_id),
        obs_i=np.int32(obs_i),
        param_names=np.array(param_names, dtype=object),
        n_chains=np.int32(n_chains),
        n_keep=np.int32(n_iter - n_burnin),
        store_thin=np.int32(store_thin),
    )
    if latent_name is not None:
        arrays["latent"] = latent[:, keep].astype(np.float32)
        arrays["latent_name"] = np.array(latent_name)
    np.savez_compressed(out_dir / f"obs_{obs_i:03d}.npz", **arrays)

    meta = {
        "scenario": scenario,
        "sampler": "sim_05_mcmc_da",
        "obs_i": int(obs_i),
        "obs_id": obs_id,
        "param_names": list(param_names),
        "n_chains": int(n_chains),
        "n_iter": int(n_iter),
        "n_burnin": int(n_burnin),
        "n_keep": int(n_iter - n_burnin),
        "store_thin": int(store_thin),
        "n_stored_per_chain": int(arrays["samples"].shape[1]),
        "n_inner_updates_per_iter": int(n_inner),
        "latent_summary": latent_name,
        "adaptation": "adaptive random walk for (log alpha, beta) and Robbins-Monro step sizes for gamma, during burn-in only",
        "latent_init": LATENT_INIT[scenario],
        "init_latent_per_chain": [r["init_latent"] for r in results],
        "diagnostics_on": "all post-burn-in draws",
        "timing": {"wall_sec": float(wall), "cpu_sec": cpu},
        "acceptance": [r["acceptance"] for r in results],
        "diagnostics": diag,
        "convergence": {"rhat": {p: diag[p]["rhat_rank"] for p in param_names}},
    }
    with open(out_dir / f"obs_{obs_i:03d}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wall={wall:.0f}s  cpu={cpu:.0f}s", flush=True)
    for name, d in diag.items():
        print(f"  {name:>22}: R-hat={d['rhat_rank']:.3f}  ESS bulk={d['ess_bulk']:.0f}  tail={d['ess_tail']:.0f}", flush=True)
    for c, r in enumerate(results):
        print(f"  chain {c} acceptance: " + "  ".join(f"{k}={v:.2f}" for k, v in r["acceptance"].items()), flush=True)


def aggregate(scenario, n_obs, draws_per_chain):
    out_dir = STRICT_ROOT / scenario / OUT_DIRNAME
    missing = [i for i in range(n_obs) if not (out_dir / f"obs_{i:03d}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} observation(s) missing in {out_dir}: {missing}")
    metas, raws, thetas, obs_idx, latents = [], [], [], [], []
    for i in range(n_obs):
        with open(out_dir / f"obs_{i:03d}_meta.json") as f:
            meta = json.load(f)
        if meta.get("sampler") != "sim_05_mcmc_da":
            raise ValueError(f"obs_{i:03d} was not produced by sim_05_mcmc_da.py")
        if scenario == "partial" and meta.get("latent_init") != LATENT_INIT[scenario]:
            raise ValueError(f"obs_{i:03d} was produced with an older latent initialisation; rerun it")
        metas.append(meta)
        d = np.load(out_dir / f"obs_{i:03d}.npz", allow_pickle=True)
        raws.append(d["samples"].astype(np.float32))
        thetas.append(d["theta"])
        obs_idx.append(int(d["obs_id"]))
        if "latent" in d.files:
            latents.append(d["latent"].astype(np.float32))
    settings = {(m["n_chains"], m["n_iter"], m["n_burnin"], m["store_thin"], m["n_inner_updates_per_iter"]) for m in metas}
    if len(settings) != 1:
        raise ValueError(f"Inconsistent run settings across observations: {settings}")
    raw = np.stack(raws)
    n_chains, n_stored = raw.shape[1], raw.shape[2]
    pick = np.unique(np.linspace(0, n_stored - 1, min(draws_per_chain, n_stored)).round().astype(int))
    compare = raw[:, :, pick, :]
    param_names = PARAM_NAMES[scenario]
    np.savez_compressed(out_dir / "mcmc_samples.npz", samples=compare, param_names=np.array(param_names, dtype=object),
                        n_chains=np.int32(n_chains), n_keep=np.int32(compare.shape[2]))
    np.savez_compressed(out_dir / "mcmc_samples_raw.npz", samples=raw, param_names=np.array(param_names, dtype=object),
                        n_chains=np.int32(n_chains), n_keep=np.int32(n_stored), store_thin=np.int32(metas[0]["store_thin"]))
    if latents:
        np.savez_compressed(out_dir / "mcmc_latent_raw.npz", latent=np.stack(latents),
                            latent_name=np.array(LATENT_NAMES[scenario]))
    np.save(out_dir / "mcmc_theta_test.npy", np.stack(thetas).astype(np.float32))
    np.save(out_dir / "mcmc_obs_idx.npy", np.array(obs_idx, dtype=np.int32))
    wall = np.array([m["timing"]["wall_sec"] for m in metas])
    cpu = np.array([m["timing"]["cpu_sec"] for m in metas])
    with open(out_dir / "mcmc_run_meta.json", "w") as f:
        json.dump({
            "scenario": scenario,
            "sampler": "sim_05_mcmc_da",
            "param_names": list(param_names),
            "n_obs": int(n_obs),
            "n_chains": int(n_chains),
            "n_iter": int(metas[0]["n_iter"]),
            "n_burnin": int(metas[0]["n_burnin"]),
            "n_keep": int(metas[0]["n_keep"]),
            "store_thin": int(metas[0]["store_thin"]),
            "n_stored_per_chain": int(n_stored),
            "n_compare_per_chain": int(compare.shape[2]),
            "n_inner_updates_per_iter": int(metas[0]["n_inner_updates_per_iter"]),
            "timing": {
                "total_wall_sec": float(wall.sum()),
                "mean_wall_per_obs": float(wall.mean()),
                "mean_cpu_per_obs": float(cpu.mean()),
            },
        }, f, indent=2)
    print(f"Aggregated {n_obs} observations in {out_dir}: stored {raw.shape}, comparison {compare.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    parser.add_argument("--obs_i", type=int, default=None)
    parser.add_argument("--n_obs", type=int, default=N_OBS)
    parser.add_argument("--n_chains", type=int, default=N_CHAINS)
    parser.add_argument("--n_iter", type=int, default=N_ITER)
    parser.add_argument("--n_burnin", type=int, default=N_BURNIN)
    parser.add_argument("--n_inner", type=int, default=None)
    parser.add_argument("--store_thin", type=int, default=STORE_THIN)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--draws_per_chain", type=int, default=COMPARE_DRAWS_PER_CHAIN)
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.scenario, args.n_obs, args.draws_per_chain)
        return
    if args.obs_i is None:
        raise ValueError("--obs_i is required unless --aggregate is given")
    if args.n_burnin >= args.n_iter:
        raise ValueError("n_burnin must be smaller than n_iter")
    n_inner = args.n_inner if args.n_inner is not None else DEFAULT_INNER[args.scenario]
    run_one_obs(args.scenario, args.obs_i, args.n_obs, args.n_chains, args.n_iter, args.n_burnin,
                n_inner, args.store_thin, N_CPUS)


if __name__ == "__main__":
    main()
