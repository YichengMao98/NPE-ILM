import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("FMD_OUT_ROOT", BASE_DIR / "output_fmd")).expanduser().resolve()
DATA_PATH = Path(os.environ.get("FMD_DATA_PATH", BASE_DIR / "fmd_clean.txt")).expanduser().resolve()
CONFIG_PATH = OUT_ROOT / "fmd_config.npz"

T_START = 30
T = 20
LATENT = 5
IP_MAX = 4

LUT_N_PTS = 800
LUT_BETA_LO = 0.01
LUT_BETA_HI = 8.0

COLUMNS = (
    "x",
    "y",
    "exposure_day",
    "cull_day",
    "z",
    "infectious_day",
    "removal_day",
    "init_inf",
)


def load_data(path):
    dat = np.loadtxt(path, dtype=np.float64)
    return {name: dat[:, j] for j, name in enumerate(COLUMNS)}


def check_data(d):
    inf = d["infectious_day"] > 0
    onset = d["infectious_day"][inf]
    if not np.all(d["exposure_day"][inf] == onset - LATENT):
        raise ValueError("Exposure day is not infectious day minus the fixed latent period for all infected farms.")
    ip = d["removal_day"][inf] - onset
    if ip.min() < 1 or ip.max() > IP_MAX:
        raise ValueError("Infectious periods fall outside 1 to IP_MAX days.")
    if not np.all(d["removal_day"][inf] == np.minimum(d["cull_day"][inf], onset + IP_MAX)):
        raise ValueError("Removal day is not min(cull day, infectious day + IP_MAX).")
    if not set(np.unique(d["z"])) <= {0.0, 1.0}:
        raise ValueError("Susceptibility covariate is not binary.")


def relative_times(d, t_start):
    infected = d["infectious_day"] > 0
    rel = lambda v: np.where(infected, v - t_start, 0).astype(np.int64)
    return infected, rel(d["exposure_day"]), rel(d["infectious_day"]), rel(d["removal_day"])


def initial_state(infected, e, onset, rem):
    active = infected & (onset <= 0) & (rem > 0)
    removed = infected & (rem <= 0)
    exposed = infected & (e <= 0) & (onset > 0)
    susceptible = ~(active | removed | exposed)
    a = np.where(active)[0].astype(np.int32)
    r = np.where(removed)[0].astype(np.int32)
    x = np.where(exposed)[0].astype(np.int32)
    return {
        "active_idx": a,
        "active_exp": e[a].astype(np.int32),
        "active_rem": rem[a].astype(np.int32),
        "removed_idx": r,
        "removed_exp": e[r].astype(np.int32),
        "removed_rem": rem[r].astype(np.int32),
        "exposed_idx": x,
        "exposed_exp": e[x].astype(np.int32),
        "susceptible_mask": susceptible,
    }


def real_observation(infected, e, onset, rem, state, T):
    n = len(e)
    window = infected & (e >= 1) & (e <= T) & state["susceptible_mask"]
    exp_obs = np.full(n, -1, dtype=np.int16)
    exp_obs[window] = e[window]
    rem_obs = np.full(n, -1, dtype=np.int16)
    has_onset = (window | np.isin(np.arange(n), state["exposed_idx"])) & (onset <= T - 1)
    rem_obs[has_onset] = np.minimum(rem[has_onset], T)
    return exp_obs, rem_obs


def empirical_ip(infected, onset, rem, T):
    ip = (rem - onset)[infected & (onset >= 1) & (onset <= T)]
    values = np.arange(1, IP_MAX + 1, dtype=np.int32)
    counts = np.array([(ip == v).sum() for v in values], dtype=np.float64)
    return values, counts / counts.sum(), counts.astype(np.int64)


def kernel_log(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    log_k = np.log1p(dist)
    np.fill_diagonal(log_k, np.inf)
    return dist, log_k


def seed_pressure(betas, log_k, active_idx, cols):
    ld = log_k[np.ix_(active_idx, cols)]
    betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    out = np.empty(len(betas))
    for s in range(0, len(betas), 100):
        e = min(s + 100, len(betas))
        out[s:e] = np.exp(-betas[s:e, None, None] * ld[None, :, :]).sum(axis=2).mean(axis=1)
    return out


def main():
    d = load_data(DATA_PATH)
    check_data(d)
    positions = np.stack([d["x"], d["y"]], axis=1)
    z = d["z"].astype(np.int8)
    n = positions.shape[0]

    infected, e, onset, rem = relative_times(d, T_START)
    state = initial_state(infected, e, onset, rem)
    exp_obs, rem_obs = real_observation(infected, e, onset, rem, state, T)
    ip_values, ip_weights, ip_counts = empirical_ip(infected, onset, rem, T)
    dist, log_k = kernel_log(positions)

    sus = np.where(state["susceptible_mask"])[0]
    sus_z1 = sus[z[sus] == 1]
    lut_betas = np.linspace(LUT_BETA_LO, LUT_BETA_HI, LUT_N_PTS)
    lut_all = seed_pressure(lut_betas, log_k, state["active_idx"], sus)
    lut_z1 = seed_pressure(lut_betas, log_k, state["active_idx"], sus_z1)

    pos_shift = positions.min(axis=0)
    region_side = float((positions.max(axis=0) - pos_shift).max())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CONFIG_PATH,
        N=np.int32(n),
        T=np.int32(T),
        t_start=np.int32(T_START),
        latent=np.int32(LATENT),
        positions=positions.astype(np.float32),
        positions_local=(positions - pos_shift).astype(np.float32),
        region_side=np.float64(region_side),
        z=z,
        log_kernel=log_k.astype(np.float32),
        ip_values=ip_values,
        ip_weights=ip_weights,
        exp_obs=exp_obs,
        rem_obs=rem_obs,
        lut_betas=lut_betas,
        lut_all=lut_all,
        lut_z1=lut_z1,
        **state,
    )

    win = exp_obs >= 1
    print(f"N={n}  window days ({T_START}, {T_START + T}]  T={T}  latent={LATENT}  IP max={IP_MAX}")
    print(f"z: low={(z == 0).sum()}  high={(z == 1).sum()}")
    print(f"initial state: infectious={len(state['active_idx'])}  exposed={len(state['exposed_idx'])}  "
          f"removed={len(state['removed_idx'])}  susceptible={int(state['susceptible_mask'].sum())}")
    print(f"exposures in window={int(win.sum())}  (z=0: {int((win & (z == 0)).sum())}, z=1: {int((win & (z == 1)).sum())})")
    print(f"onsets in window={int((rem_obs >= 0).sum())}  removals in window={int(((rem_obs >= 1) & (rem_obs < T)).sum())}")
    print(f"empirical IP from farms with onset in days ({T_START}, {T_START + T}]: counts {dict(zip(ip_values.tolist(), ip_counts.tolist()))}  "
          f"weights={np.round(ip_weights, 4).tolist()}  mean={float(np.dot(ip_values, ip_weights)):.3f}")
    for b in (1.0, 2.0, 3.0):
        print(f"beta={b}: lambda0 all={np.interp(b, lut_betas, lut_all):.3f}  z1={np.interp(b, lut_betas, lut_z1):.3f}")
    print(f"Saved: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
