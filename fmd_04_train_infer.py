import argparse
import json
import time

import numpy as np
import torch

from fmd_03_tune_npe import (
    DATA_ROOT,
    DEVICE,
    FIXED_HPARAMS,
    NPE_ROOT,
    PARAM_NAMES,
    WeightedGNN,
    FMDPrior,
    build_cnn,
    build_gnn,
    cnn_norm_stats,
    encode_cnn,
    encode_gnn,
    load_config,
    load_split,
    sample_valid,
    set_seed,
    train_npe,
)

N_POST = 3000

HP_KEYS = {
    "cnn": ("learning_rate", "num_transforms", "emb_dim", "channels1", "channels2", "hidden_features"),
    "gnn": ("learning_rate", "num_transforms", "emb_dim", "gnn_hidden", "gnn_layers", "gnn_k", "hidden_features"),
}


def count_params(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def final_hparams(saved, kind, seed):
    best = saved[kind]["best_config"]
    hps = {**FIXED_HPARAMS, "seed": int(seed)}
    for k in HP_KEYS[kind]:
        hps[k] = float(best[k]) if k == "learning_rate" else int(best[k])
    return hps


def fit_and_infer(kind, emb, x_tr, th_tr, x_real, prior, hps, seed, out_dir):
    label = "CNN-NPE" if kind == "cnn" else "GNN-NPE (distance-weighted)"
    n_emb = count_params(emb)
    print(f"\n{label}  seed={seed}  embedding params={n_emb}  hps={hps}", flush=True)

    posterior, de, info = train_npe(th_tr, x_tr, prior, emb.to(DEVICE), hps)
    print(f"  trained: epochs={info.get('epochs_run', '?')}  best_epoch={info.get('best_epoch', '?')}  "
          f"{info['train_sec']:.0f}s", flush=True)
    torch.save(de, out_dir / f"{kind}_density_estimator.pt")

    set_seed(seed)
    xo = torch.tensor(x_real, dtype=torch.float32, device=DEVICE)
    t0 = time.perf_counter()
    samples = sample_valid(posterior, xo, N_POST)
    infer_sec = time.perf_counter() - t0
    if samples is None:
        raise RuntimeError(f"{label}: fewer than {N_POST} valid posterior samples for the observed data")

    np.savez_compressed(
        out_dir / f"{kind}_real_posterior.npz",
        samples=samples.astype(np.float32),
        param_names=np.array(PARAM_NAMES, dtype=object),
    )
    lambdas = emb.lambdas() if isinstance(emb, WeightedGNN) else []
    meta = {
        "model": label,
        "seed": int(seed),
        "param_names": list(PARAM_NAMES),
        "n_post_samples": N_POST,
        "n_params_embedding": n_emb,
        "n_params_total": count_params(de),
        "learned_lambda_per_layer": lambdas,
        "timing": {"train_sec": info["train_sec"], "infer_sec": float(infer_sec)},
        "training": {**{k: v for k, v in info.items() if k != "train_sec"}, "n_train": int(th_tr.shape[0])},
        "hparams": hps,
        "z_score_x": "none",
    }
    with open(out_dir / f"{kind}_run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    med = np.median(samples, axis=0)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    for j, p in enumerate(PARAM_NAMES):
        print(f"  {p:>8}: median={med[j]:.5g}  95% CI=[{lo[j]:.5g}, {hi[j]:.5g}]", flush=True)
    if lambdas:
        print(f"  learned lambda per layer: {[round(v, 4) for v in lambdas]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=["cnn", "gnn"], choices=["cnn", "gnn"])
    args = parser.parse_args()

    seed = int(args.seed)
    cfg = load_config()
    prior = FMDPrior(cfg)
    with open(NPE_ROOT / "best_hparams.json") as f:
        saved = json.load(f)

    th_tr, exp_tr, rem_tr = load_split(DATA_ROOT / "final" / "train.npz")
    exp_obs, rem_obs = cfg["exp_obs"], cfg["rem_obs"]
    out_dir = NPE_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed={seed}  models={args.models}  n_train={th_tr.shape[0]}  N_POST={N_POST}  device={DEVICE}", flush=True)

    if "cnn" in args.models:
        mean, std = cnn_norm_stats(exp_tr, rem_tr, cfg)
        np.savez_compressed(out_dir / "cnn_norm.npz", mean=mean, std=std)
        hps = final_hparams(saved, "cnn", seed)
        set_seed(seed)
        emb = build_cnn(hps)
        fit_and_infer(
            "cnn",
            emb,
            encode_cnn(exp_tr, rem_tr, cfg, mean, std),
            th_tr,
            encode_cnn(exp_obs, rem_obs, cfg, mean, std),
            prior,
            hps,
            seed,
            out_dir,
        )

    if "gnn" in args.models:
        hps = final_hparams(saved, "gnn", seed)
        set_seed(seed)
        emb = build_gnn(hps, cfg, weighted=True)
        fit_and_infer(
            "gnn",
            emb,
            encode_gnn(exp_tr, rem_tr, cfg),
            th_tr,
            encode_gnn(exp_obs, rem_obs, cfg),
            prior,
            hps,
            seed,
            out_dir,
        )

    print(f"\nOutputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
