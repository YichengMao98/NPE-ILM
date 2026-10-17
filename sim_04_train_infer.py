import argparse
import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn

from sim_03_tune_npe import (
    CNN1D,
    WeightedGNN,
    DEVICE,
    SEED,
    FIXED_HPARAMS,
    load_config,
    load_cnn,
    load_gnn,
    make_prior,
    set_seed,
    valid_mask_by_scenario,
)

BASE_FILE_DIR = Path(__file__).resolve().parent
ARC_ROOT = Path(os.environ.get("ILM_LOCAL_ROOT", BASE_FILE_DIR / "output")).expanduser().resolve()
STRICT_ROOT = Path(os.environ.get("ILM_STRICT_ROOT", ARC_ROOT / "strict")).expanduser().resolve()
SEEDS_ROOT = ARC_ROOT / "strict_seeds"

N_POST = 3000

PARAM_NAMES = {
    "full": ("alpha", "beta"),
    "partial": ("alpha", "beta", "rho"),
    "gamma": ("alpha", "beta", "gamma"),
}


def sample_valid_posterior_strict(
    posterior,
    x_obs,
    scenario,
    lut_betas,
    lut_seed_rowsums,
    log_dist,
    seed_idx,
    n_post,
):
    batch_draws = max(4000, 4 * n_post)
    kept = []
    tries = 0
    max_tries = 50

    while sum(arr.shape[0] for arr in kept) < n_post and tries < max_tries:
        tries += 1
        raw = posterior.sample((batch_draws,), x=x_obs, show_progress_bars=False).detach().cpu().numpy()
        mask = valid_mask_by_scenario(raw, scenario, lut_betas, lut_seed_rowsums, log_dist, seed_idx)
        good = raw[mask]

        if good.shape[0] > 0:
            kept.append(good.astype(np.float32))

    n_kept = sum(arr.shape[0] for arr in kept)

    if n_kept < n_post:
        raise RuntimeError(
            f"Only {n_kept}/{n_post} valid posterior samples collected for scenario={scenario}."
        )

    return np.concatenate(kept, axis=0)[:n_post]


def train_npe(x_tr, th_tr, hps, prior, emb, device):
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


def run_inference(
    posterior,
    x_te,
    n_params,
    scenario,
    lut_betas,
    lut_seed_rowsums,
    log_dist,
    seed_idx,
    device,
):
    n = x_te.shape[0]
    samples = np.empty((n, N_POST, n_params), dtype=np.float32)

    t0 = time.perf_counter()

    for i in range(n):
        x_obs = torch.tensor(x_te[i:i + 1], dtype=torch.float32, device=device)
        samples[i] = sample_valid_posterior_strict(
            posterior,
            x_obs,
            scenario,
            lut_betas,
            lut_seed_rowsums,
            log_dist,
            seed_idx,
            N_POST,
        )

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{n}  {elapsed:.0f}s  ({elapsed / (i + 1) * 1000:.1f} ms/obs)")

    total = time.perf_counter() - t0

    return samples, total, total / n * 1000


def count_params(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def density_estimator(posterior):
    for name in ("posterior_estimator", "net", "_neural_net"):
        net = getattr(posterior, name, None)
        if isinstance(net, nn.Module):
            return net
    return None


def prepare_eval_layout(scenario, out_scenario_dir, final_dir):
    src_final = STRICT_ROOT / scenario / "final"
    shutil.copy2(src_final / "test.npz", final_dir / "test.npz")
    mcmc_link = out_scenario_dir / "mcmc"
    src_mcmc = STRICT_ROOT / scenario / "mcmc"
    if mcmc_link.is_symlink() and not mcmc_link.exists():
        mcmc_link.unlink()
    if src_mcmc.exists() and not mcmc_link.is_symlink() and not mcmc_link.exists():
        os.symlink(os.path.relpath(src_mcmc, out_scenario_dir), mcmc_link, target_is_directory=True)


def fit_and_save(tag, label, emb, x_tr, th_tr, x_te, th_te, hps, prior, cfg, scenario,
                 param_names, seed, out_dir):
    n_params = len(param_names)
    n_emb = count_params(emb)

    print(f"Training {label} on {th_tr.shape[0]} simulations (embedding params={n_emb})...")

    post, info = train_npe(x_tr, th_tr, hps, prior, emb, DEVICE)

    print(
        f"  done: epochs_run={info.get('epochs_run', '?')}  "
        f"best_epoch={info.get('best_epoch', '?')}  {info['train_sec']:.0f}s"
    )

    net = density_estimator(post)
    n_total = count_params(net) if net is not None else None
    if net is not None:
        torch.save(net.state_dict(), out_dir / f"{tag}_density_estimator_state.pt")

    print(f"Running {label} inference on {th_te.shape[0]} observations (N_POST={N_POST})...")

    set_seed(seed)
    samp, isec, mpo = run_inference(
        post,
        x_te,
        n_params,
        scenario,
        cfg["lut_betas"],
        cfg["lut_seed_rowsums"],
        cfg["log_dist"],
        cfg["seed_idx"],
        DEVICE,
    )

    np.savez_compressed(
        out_dir / f"{tag}_posterior_samples.npz",
        samples=samp,
        param_names=np.array(param_names, dtype=object),
        n_post_samples=np.int32(N_POST),
    )

    lambdas = emb.lambdas() if isinstance(emb, WeightedGNN) else []

    with open(out_dir / f"{tag}_run_meta.json", "w") as f:
        json.dump(
            {
                "model": label,
                "scenario": scenario,
                "seed": int(seed),
                "param_names": list(param_names),
                "n_post_samples": int(N_POST),
                "n_params_embedding": n_emb,
                "n_params_total": n_total,
                "learned_lambda_per_layer": lambdas,
                "timing": {
                    "train_sec": info["train_sec"],
                    "infer_sec": isec,
                    "ms_per_obs": mpo,
                },
                "training": {
                    "epochs_run": info.get("epochs_run", 0),
                    "best_epoch": info.get("best_epoch", -1),
                    "best_val_lp": info.get("best_val_lp", float("nan")),
                    "n_train": int(th_tr.shape[0]),
                    "n_test": int(th_te.shape[0]),
                },
                "hparams": hps,
            },
            f,
            indent=2,
        )

    print(f"Saved {tag}_posterior_samples.npz  shape={samp.shape}  ({mpo:.1f} ms/obs)")
    if lambdas:
        print(f"  learned lambda per layer: {[round(v, 4) for v in lambdas]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["full", "partial", "gamma"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", nargs="+", default=["cnn", "gnn"], choices=["cnn", "gnn"])
    args = parser.parse_args()

    scenario = args.scenario
    seed = int(args.seed)
    param_names = PARAM_NAMES[scenario]

    cfg = load_config()
    prior = make_prior(scenario, cfg["lut_betas"], cfg["lut_seed_rowsums"], cfg["log_dist"], cfg["seed_idx"])

    scenario_dir = STRICT_ROOT / scenario
    src_final_dir = scenario_dir / "final"
    out_scenario_dir = SEEDS_ROOT / f"seed_{seed}" / scenario
    out_dir = out_scenario_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    hp_path = scenario_dir / "best_hparams.json"

    if not hp_path.exists():
        raise FileNotFoundError(f"best_hparams.json not found: {hp_path}\nRun sim_03 first.")

    with open(hp_path) as f:
        saved = json.load(f)

    stored_names = [str(v) for v in np.load(src_final_dir / "test.npz", allow_pickle=True)["param_names"]]
    if stored_names != list(param_names):
        raise ValueError(f"Parameter order mismatch: data={stored_names} expected={list(param_names)}")

    print(f"scenario={scenario}  seed={seed}  models={args.models}  params={param_names}  device={DEVICE}")
    print(
        f"N_POST={N_POST}  max_epochs={FIXED_HPARAMS['max_num_epochs']}  "
        f"early_stop={FIXED_HPARAMS['stop_after_epochs']}"
    )
    print(f"Output: {out_dir}")

    th_tr, x_cnn_tr, th_te, x_cnn_te, T_len = load_cnn(src_final_dir)
    th_tr2, x_gnn_tr, th_te2, x_gnn_te, _ = load_gnn(src_final_dir)

    if not np.allclose(th_tr, th_tr2):
        raise ValueError("CNN and GNN training theta arrays do not match.")

    if not np.allclose(th_te, th_te2):
        raise ValueError("CNN and GNN test theta arrays do not match.")

    print(f"Final: n_train={th_tr.shape[0]}  n_test={th_te.shape[0]}  T={T_len}")

    np.save(out_dir / "theta_test.npy", th_te)
    prepare_eval_layout(scenario, out_scenario_dir, out_dir)

    if "cnn" in args.models:
        print(f"\n{'=' * 60}\nCNN-NPE")

        cnn_best = saved["cnn"]["best_config"]
        cnn_hps = {
            **FIXED_HPARAMS,
            "seed": seed,
            "learning_rate": float(cnn_best["learning_rate"]),
            "num_transforms": int(cnn_best["num_transforms"]),
            "emb_dim": int(cnn_best["emb_dim"]),
            "channels1": int(cnn_best["channels1"]),
            "channels2": int(cnn_best["channels2"]),
            "hidden_features": int(cnn_best.get("hidden_features", 64)),
        }

        print(f"Hyperparams: {cnn_best}")

        set_seed(seed)
        cnn_emb = CNN1D(
            T_len,
            emb_dim=cnn_hps["emb_dim"],
            channels1=cnn_hps["channels1"],
            channels2=cnn_hps["channels2"],
            pool_out=cnn_hps["pool_out"],
            dropout=cnn_hps["dropout"],
        ).to(DEVICE)

        fit_and_save("cnn", "CNN-NPE", cnn_emb, x_cnn_tr, th_tr, x_cnn_te, th_te, cnn_hps,
                     prior, cfg, scenario, param_names, seed, out_dir)

    if "gnn" in args.models:
        print(f"\n{'=' * 60}\nGNN-NPE (distance-weighted)")

        gnn_best = saved["gnn"]["best_config"]
        gnn_hps = {
            **FIXED_HPARAMS,
            "seed": seed,
            "learning_rate": float(gnn_best["learning_rate"]),
            "num_transforms": int(gnn_best["num_transforms"]),
            "emb_dim": int(gnn_best["emb_dim"]),
            "gnn_hidden": int(gnn_best["gnn_hidden"]),
            "gnn_layers": int(gnn_best["gnn_layers"]),
            "gnn_k": int(gnn_best["gnn_k"]),
            "hidden_features": int(gnn_best.get("hidden_features", 64)),
        }

        print(f"Hyperparams: {gnn_best}")

        set_seed(seed)
        gnn_emb = WeightedGNN(
            cfg["positions"],
            cfg["region_side"],
            T_len,
            emb_dim=gnn_hps["emb_dim"],
            hidden=gnn_hps["gnn_hidden"],
            n_layers=gnn_hps["gnn_layers"],
            k=gnn_hps["gnn_k"],
            dropout=gnn_hps["dropout"],
        ).to(DEVICE)

        fit_and_save("gnn", "GNN-NPE", gnn_emb, x_gnn_tr, th_tr2, x_gnn_te, th_te2, gnn_hps,
                     prior, cfg, scenario, param_names, seed, out_dir)

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
