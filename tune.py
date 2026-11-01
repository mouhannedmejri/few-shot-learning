"""
tune.py — val-only hyperparameter search.

Runs 100 val episodes per (dataset, encoder, shots) combination and
selects best tau, lam, alpha (TIM) and C_reg (logreg) by mean accuracy.
Writes configs/best_hparams.json on completion.

Usage:
  python tune.py --dataset bloodmnist --encoders resnet50 dinov2_s --episodes 100
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fewshot import (
    Episode,
    aggregate_episodes,
    compute_metrics,
    predict_logreg,
    predict_proto,
    predict_tim,
    sample_episode,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("features")
CONFIG_DIR = Path("configs")

# Hyperparameter grids (balanced val only — see CLAUDE.md)
TAU_GRID = [5, 10, 20, 30]
LAM_GRID = [0.1, 0.3, 0.5, 1.0, 2.0]
ALPHA_GRID = [0.5, 1.0]
C_REG_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SHOTS_LIST = [1, 5, 10, 20]


TUNING_SPLITS = ("pool", "val")


def load_features(dataset: str, encoder: str, split: str, feature_dir: Path = FEATURE_DIR):
    """Load (X, Y) npy pair for given dataset/encoder/split. Refuses the test split."""
    if split not in TUNING_SPLITS:
        raise ValueError(f"tune.py may only load {TUNING_SPLITS}, not {split!r} (test leakage)")
    prefix = feature_dir / f"{dataset}_{encoder}"
    X = np.load(f"{prefix}_{split}_X.npy")
    Y = np.load(f"{prefix}_{split}_Y.npy")
    return X, Y


def run_episodes(
    pool_X, pool_y, val_X, val_y,
    n_classes: int,
    shots: int,
    n_episodes: int,
    hparams: dict,
    method: str,
    base_seed: int = 1000,
) -> list[dict]:
    """Run n_episodes and return per-episode metric dicts."""
    n_query_total = n_classes * 30
    rows: list[dict] = []
    for ep_id in range(n_episodes):
        ep = sample_episode(
            pool_X=pool_X, pool_y=pool_y,
            query_X=val_X, query_y=val_y,
            n_classes=n_classes,
            shots=shots,
            n_query_total=n_query_total,
            episode_id=ep_id,
            support_seed=base_seed + ep_id,
            imbalance=0.0,
        )
        if method == "logreg":
            preds = predict_logreg(ep, C_reg=hparams.get("C_reg", 1.0))
        elif method == "tim":
            preds = predict_tim(
                ep,
                tau=hparams.get("tau", 20.0),
                lam=hparams.get("lam", 0.5),
                alpha=hparams.get("alpha", 1.0),
            )
        else:
            raise ValueError(f"tune.py only tunes logreg and tim, not {method}")
        m = compute_metrics(ep.query_y, preds, n_classes=n_classes)
        rows.append(m)
    return rows


def tune_logreg(pool_X, pool_y, val_X, val_y, n_classes, shots, n_episodes, base_seed):
    best_acc = -1.0
    best_C = 1.0
    for C_reg in C_REG_GRID:
        rows = run_episodes(
            pool_X, pool_y, val_X, val_y, n_classes, shots, n_episodes,
            hparams={"C_reg": C_reg}, method="logreg", base_seed=base_seed,
        )
        acc = float(np.mean([r["acc"] for r in rows]))
        logger.debug("  logreg C=%.3f acc=%.4f", C_reg, acc)
        if acc > best_acc:
            best_acc = acc
            best_C = C_reg
    return {"C_reg": best_C, "val_acc": best_acc}


def tune_tim(pool_X, pool_y, val_X, val_y, n_classes, shots, n_episodes, base_seed):
    best_acc = -1.0
    best_params = {"tau": 20.0, "lam": 0.5, "alpha": 1.0}
    for tau, lam, alpha in itertools.product(TAU_GRID, LAM_GRID, ALPHA_GRID):
        rows = run_episodes(
            pool_X, pool_y, val_X, val_y, n_classes, shots, n_episodes,
            hparams={"tau": tau, "lam": lam, "alpha": alpha},
            method="tim", base_seed=base_seed,
        )
        acc = float(np.mean([r["acc"] for r in rows]))
        logger.debug("  tim tau=%d lam=%.2f alpha=%.2f acc=%.4f", tau, lam, alpha, acc)
        # Ties: prefer smaller lam
        if acc > best_acc or (acc == best_acc and lam < best_params["lam"]):
            best_acc = acc
            best_params = {"tau": float(tau), "lam": float(lam), "alpha": float(alpha)}
    return {**best_params, "val_acc": best_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only hyperparameter tuning.")
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument("--encoders", nargs="+", default=["resnet50", "dinov2_s", "clip_b16"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--shots", nargs="+", type=int, default=SHOTS_LIST)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run 2 episodes per config for quick validation."
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    n_eps = 2 if args.smoke else args.episodes

    best: dict = {
        "dataset": args.dataset,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "smoke": args.smoke,
        "n_val_episodes": n_eps,
        "val_support_base_seed": args.base_seed,
        "encoders": {},
    }

    for enc_key in args.encoders:
        logger.info("=== Encoder: %s ===", enc_key)
        try:
            pool_X, pool_y = load_features(args.dataset, enc_key, "pool", feature_dir)
            val_X, val_y = load_features(args.dataset, enc_key, "val", feature_dir)
        except FileNotFoundError as e:
            logger.error("Feature files not found: %s. Run extract_features.py first.", e)
            continue

        n_classes = len(np.unique(pool_y))
        logger.info("n_classes=%d, pool=%d, val=%d", n_classes, len(pool_y), len(val_y))

        enc_best: dict = {}
        for shots in args.shots:
            logger.info("  shots=%d", shots)
            t0 = time.time()
            lr_result = tune_logreg(
                pool_X, pool_y, val_X, val_y, n_classes, shots, n_eps, args.base_seed
            )
            tim_result = tune_tim(
                pool_X, pool_y, val_X, val_y, n_classes, shots, n_eps, args.base_seed
            )
            elapsed = time.time() - t0
            logger.info(
                "  shots=%d logreg best C=%.3f (val_acc=%.4f) | "
                "tim best tau=%.1f lam=%.2f alpha=%.2f (val_acc=%.4f) [%.1fs]",
                shots,
                lr_result["C_reg"], lr_result["val_acc"],
                tim_result["tau"], tim_result["lam"], tim_result["alpha"],
                tim_result["val_acc"],
                elapsed,
            )
            enc_best[str(shots)] = {
                "logreg": lr_result,
                "tim": tim_result,
                # tim_cond uses same tau/alpha; lam can be re-tuned in ablation
                "tim_cond": {"tau": tim_result["tau"], "lam": tim_result["lam"],
                             "alpha": tim_result["alpha"]},
            }
        best["encoders"][enc_key] = enc_best

    out_path = config_dir / "best_hparams.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Saved hyperparameters to %s", out_path)


if __name__ == "__main__":
    main()
