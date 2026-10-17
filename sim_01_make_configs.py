import os
from pathlib import Path

import numpy as np

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
CONFIG_ROOT = Path(os.environ.get("ILM_CONFIG_ROOT", ARC_ROOT / "configs")).expanduser().resolve()
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

N = 500
REGION_SIDE = 100.0
I0 = 3
BASE_SEED = 2026

LUT_N_BETAS = 600
BETA_MIN = 0.2
BETA_MAX = 4.0


def make_positions(rng):
    return rng.uniform(0.0, REGION_SIDE, size=(N, 2)).astype(np.float32)


def make_log_dist(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float64)
    log_dist = np.zeros_like(dist)
    mask = dist > 0
    log_dist[mask] = np.log(dist[mask])
    return log_dist.astype(np.float32)


def make_seed_idx(rng):
    return rng.choice(N, size=I0, replace=False).astype(np.int32)


def build_lut(log_dist, seed_idx):
    betas = np.linspace(BETA_MIN, BETA_MAX, LUT_N_BETAS)
    seed_rowsums = np.empty(LUT_N_BETAS, dtype=np.float64)
    for k, b in enumerate(betas):
        W = np.exp(-b * log_dist.astype(np.float64))
        np.fill_diagonal(W, 0.0)
        seed_rowsums[k] = W[seed_idx, :].sum(axis=1).mean()
    return betas, seed_rowsums


def main():
    rng = np.random.default_rng(BASE_SEED)
    positions = make_positions(rng)
    log_dist = make_log_dist(positions)
    seed_idx = make_seed_idx(rng)
    lut_betas, lut_seed_rowsums = build_lut(log_dist, seed_idx)

    out_path = CONFIG_ROOT / "ilm_config.npz"
    np.savez(
        out_path,
        N=np.int32(N),
        region_side=np.float64(REGION_SIDE),
        positions=positions,
        log_dist=log_dist,
        seed_idx=seed_idx,
        lut_betas=lut_betas,
        lut_seed_rowsums=lut_seed_rowsums,
    )

    print(f"Saved: {out_path}")
    print(f"N={N} region={REGION_SIDE}x{REGION_SIDE} I0={I0}")
    print(f"seed_idx={seed_idx.tolist()}")
    print(f"LUT beta in [{BETA_MIN}, {BETA_MAX}], n_grid={LUT_N_BETAS}")
    sr_mid = float(np.interp(1.5, lut_betas, lut_seed_rowsums))
    print(f"seed_rowsum @ beta=1.5: {sr_mid:.4f}")


if __name__ == "__main__":
    main()