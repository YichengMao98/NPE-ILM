import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("FMD_OUT_ROOT", BASE_DIR / "output_fmd")).expanduser().resolve()
CONFIG_PATH = OUT_ROOT / "fmd_config.npz"
DATA_ROOT = OUT_ROOT / "data"

N_CPUS = max(1, int(os.environ.get("FMD_N_CPUS", os.cpu_count() or 1)))

BASE_SEED = 1234
CHUNK = 250

BETA_SHAPE = 6.0
BETA_SCALE = 1.0 / 4.0
R0_SHAPE = 10.0
R0_SCALE = 1.0 / 4.0

PARAM_NAMES = ("alpha0", "alpha1", "beta")

UNEXPOSED = np.int32(-(10**6))

SPLITS = {
    "pilot": (("train", 3000, 0), ("test", 300, 1)),
    "final": (("train", 20000, 100),),
    "reference": (("reference", 10000, 200),),
}

_CFG = None


def load_config(path=CONFIG_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}. Run fmd_01 first.")
    dat = np.load(path, allow_pickle=True)
    return {k: dat[k] for k in dat.files}


def mean_ip(cfg):
    return float(np.dot(cfg["ip_values"], cfg["ip_weights"]))


def lambda0(beta, r, cfg):
    beta = float(beta)
    lut_b = cfg["lut_betas"]
    if lut_b[0] <= beta <= lut_b[-1]:
        l_all = float(np.interp(beta, lut_b, cfg["lut_all"]))
        l_z1 = float(np.interp(beta, lut_b, cfg["lut_z1"]))
    else:
        sus = np.where(cfg["susceptible_mask"])[0]
        lk = cfg["log_kernel"][np.ix_(cfg["active_idx"], sus)].astype(np.float64)
        k = np.exp(-beta * lk)
        l_all = float(k.sum(axis=1).mean())
        l_z1 = float(k[:, cfg["z"][sus] == 1].sum(axis=1).mean())
    return r * l_all + (1.0 - r) * l_z1


def theta_from_natural(r0, beta, r, cfg):
    a_high = r0 / (mean_ip(cfg) * lambda0(beta, r, cfg))
    return np.array([r * a_high, (1.0 - r) * a_high, beta], dtype=np.float64)


def draw_prior(rng, cfg):
    beta = float(rng.gamma(BETA_SHAPE, BETA_SCALE))
    r0 = float(rng.gamma(R0_SHAPE, R0_SCALE))
    r = float(rng.uniform(0.0, 1.0))
    return theta_from_natural(r0, beta, r, cfg), r0, r


def initial_arrays(cfg, rng):
    n = int(cfg["N"])
    L = int(cfg["latent"])
    exp_t = np.full(n, UNEXPOSED, dtype=np.int32)
    onset = np.full(n, 10**6, dtype=np.int32)
    rem = np.full(n, 10**6, dtype=np.int32)
    for key in ("active", "removed"):
        idx = cfg[f"{key}_idx"]
        exp_t[idx] = cfg[f"{key}_exp"]
        onset[idx] = cfg[f"{key}_exp"] + L
        rem[idx] = cfg[f"{key}_rem"]
    idx = cfg["exposed_idx"]
    exp_t[idx] = cfg["exposed_exp"]
    onset[idx] = cfg["exposed_exp"] + L
    rem[idx] = onset[idx] + rng.choice(cfg["ip_values"], size=idx.size, p=cfg["ip_weights"])
    return exp_t, onset, rem


def simulate(theta, cfg, rng, log_kernel=None):
    alpha0, alpha1, beta = (float(v) for v in theta)
    T = int(cfg["T"])
    L = int(cfg["latent"])
    if log_kernel is None:
        log_kernel = cfg["log_kernel"].astype(np.float64)
    W = np.exp(-beta * log_kernel)
    coef = alpha0 + alpha1 * cfg["z"].astype(np.float64)
    exp_t, onset, rem = initial_arrays(cfg, rng)

    for t in range(T):
        sus = np.where(exp_t == UNEXPOSED)[0]
        infectious = np.where((onset <= t) & (rem > t))[0]
        if sus.size == 0 or infectious.size == 0:
            continue
        lam = coef[sus] * W[np.ix_(sus, infectious)].sum(axis=1)
        hit = sus[rng.random(sus.size) < -np.expm1(-lam)]
        if hit.size:
            exp_t[hit] = t + 1
            onset[hit] = t + 1 + L
            rem[hit] = onset[hit] + rng.choice(cfg["ip_values"], size=hit.size, p=cfg["ip_weights"])

    window = (exp_t >= 1) & (exp_t <= T)
    exp_out = np.where(window, exp_t, -1).astype(np.int16)
    has_onset = (window | np.isin(np.arange(exp_t.size), cfg["exposed_idx"])) & (onset <= T - 1)
    rem_out = np.where(has_onset, np.minimum(rem, T), -1).astype(np.int16)
    return exp_out, rem_out


def _init_worker(config_path):
    global _CFG
    _CFG = load_config(Path(config_path))
    _CFG["log_kernel"] = _CFG["log_kernel"].astype(np.float64)


def _run_chunk(args):
    n, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    N = int(_CFG["N"])
    theta = np.empty((n, len(PARAM_NAMES)), dtype=np.float32)
    exp = np.empty((n, N), dtype=np.int16)
    rem = np.empty((n, N), dtype=np.int16)
    nat = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        th, r0, r = draw_prior(rng, _CFG)
        theta[i] = th
        nat[i] = (r0, r)
        exp[i], rem[i] = simulate(th, _CFG, rng, _CFG["log_kernel"])
    return theta, exp, rem, nat


def generate(n, seed, n_procs):
    sizes = [CHUNK] * (n // CHUNK) + ([n % CHUNK] if n % CHUNK else [])
    seqs = np.random.SeedSequence(seed).spawn(len(sizes))
    with Pool(processes=n_procs, initializer=_init_worker, initargs=(str(CONFIG_PATH),)) as pool:
        parts = pool.map(_run_chunk, list(zip(sizes, seqs)))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def save_split(out_dir, name, theta, exp, rem, nat, cfg):
    T = int(cfg["T"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{name}.npz",
        theta=theta,
        exp=exp,
        rem=rem,
        T=np.int32(T),
        param_names=np.array(PARAM_NAMES, dtype=object),
    )
    z = cfg["z"].astype(bool)
    stats = pd.DataFrame(theta, columns=list(PARAM_NAMES))
    stats["r0"] = nat[:, 0]
    stats["r"] = nat[:, 1]
    stats["n_exposures"] = (exp >= 1).sum(axis=1)
    stats["n_exposures_z0"] = ((exp >= 1) & ~z).sum(axis=1)
    stats["n_exposures_z1"] = ((exp >= 1) & z).sum(axis=1)
    stats["n_removals"] = ((rem >= 1) & (rem < T)).sum(axis=1)
    stats.to_csv(out_dir / f"{name}_stats.csv", index=False)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["final", "reference"], choices=list(SPLITS))
    parser.add_argument("--n_procs", type=int, default=N_CPUS)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config()
    n_real = int((cfg["exp_obs"] >= 1).sum())
    print(f"N={int(cfg['N'])}  T={int(cfg['T'])}  latent={int(cfg['latent'])}  E[IP]={mean_ip(cfg):.3f}  procs={args.n_procs}")
    print(f"prior: beta~Gamma({BETA_SHAPE}, {BETA_SCALE:.4g})  R0~Gamma({R0_SHAPE}, {R0_SCALE:.4g})  r~Uniform(0, 1)")

    for set_name in args.sets:
        for split, n, offset in SPLITS[set_name]:
            n_use = max(1, int(round(n * args.scale)))
            t0 = time.perf_counter()
            theta, exp, rem, nat = generate(n_use, BASE_SEED + offset, args.n_procs)
            stats = save_split(DATA_ROOT / set_name, split, theta, exp, rem, nat, cfg)
            ne = stats["n_exposures"]
            q = float((ne < n_real).mean() + 0.5 * (ne == n_real).mean())
            print(f"[{set_name}/{split}] n={n_use}  {time.perf_counter() - t0:.1f}s  exposures median={ne.median():.0f} "
                  f"IQR=[{ne.quantile(0.25):.0f}, {ne.quantile(0.75):.0f}]  real={n_real} (prior predictive quantile {q:.3f})",
                  flush=True)
    print(f"Saved under {DATA_ROOT}")


if __name__ == "__main__":
    main()
