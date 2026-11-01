"""
run_experiments.py — main experiment loop.

Loops over encoder x method x shots x imbalance x episodes and appends
rows to results/results.csv (long format).

Usage:
  python run_experiments.py --dataset bloodmnist --final --episodes 300
  python run_experiments.py --smoke   # uses synthetic features, finishes in seconds
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from fewshot import (
    compute_metrics,
    predict_logreg,
    predict_majority,
    predict_proto,
    predict_random,
    predict_tim,
    predict_tim_cond,
    sample_episode,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("features")
RESULTS_DIR = Path("results")
CONFIG_DIR = Path("configs")
SMOKE_DIR = Path("smoke")  # gitignored; smoke runs never touch the real dirs

CSV_COLUMNS = [
    "dataset", "encoder", "method", "shots", "imbalance", "episode",
    "split", "n_query", "acc", "macro_f1", "bal_acc",
]

SHOTS_LIST = [1, 5, 10, 20]
IMBALANCE_LEVELS = [0.0, 1.0, 0.3]  # 0.0 = balanced
METHODS = ["proto", "logreg", "tim", "tim_cond", "random", "majority"]


def load_features(dataset: str, encoder: str, split: str, feature_dir: Path = FEATURE_DIR):
    prefix = feature_dir / f"{dataset}_{encoder}"
    X = np.load(f"{prefix}_{split}_X.npy")
    Y = np.load(f"{prefix}_{split}_Y.npy")
    return X, Y


def load_hparams(hparams_path: Path, encoder: str, shots: int, method: str) -> dict:
    with open(hparams_path) as f:
        hp = json.load(f)
    enc_hp = hp.get("encoders", {}).get(encoder, {})
    shot_hp = enc_hp.get(str(shots), {})
    return shot_hp.get(method, {})


def validate_final_hparams(hparams_path: Path) -> str | None:
    """Return an error message if the config is unfit for a final test run, else None."""
    if not hparams_path.exists():
        return f"{hparams_path} not found. Run tune.py first."
    with open(hparams_path) as f:
        hp = json.load(f)
    if hp.get("timestamp") == "smoke" or hp.get("smoke"):
        return f"{hparams_path} was written by a smoke run. Run tune.py on real features."
    if "val_support_base_seed" not in hp or not hp.get("encoders"):
        return f"{hparams_path} lacks tune.py metadata (val seeds / encoders)."
    return None


def row_key(row: dict) -> tuple:
    """Identity of a results row: one per (dataset, encoder, method, shots, imbalance, episode)."""
    return (
        str(row["dataset"]), str(row["encoder"]), str(row["method"]),
        int(row["shots"]), float(row["imbalance"]), int(row["episode"]),
    )


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def prepare_csv(csv_path: Path, planned_keys: set[tuple], force: bool) -> None:
    """Refuse to duplicate rows already in csv_path; with force, drop them first."""
    existing = read_existing_rows(csv_path)
    clashes = [r for r in existing if row_key(r) in planned_keys]
    if not clashes:
        return
    if not force:
        raise SystemExit(
            f"{csv_path} already has {len(clashes)} of the planned rows. "
            "Use --force to replace them, or --out for a new file."
        )
    kept = [r for r in existing if row_key(r) not in planned_keys]
    logger.warning("--force: replacing %d existing rows in %s", len(clashes), csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)


def get_method_preds(method: str, ep, hparams: dict) -> np.ndarray:
    if method == "proto":
        return predict_proto(ep)
    elif method == "logreg":
        C_reg = float(hparams.get("C_reg", 1.0))
        return predict_logreg(ep, C_reg=C_reg)
    elif method == "tim":
        return predict_tim(
            ep,
            tau=float(hparams.get("tau", 20.0)),
            lam=float(hparams.get("lam", 0.5)),
            alpha=float(hparams.get("alpha", 1.0)),
        )
    elif method == "tim_cond":
        return predict_tim_cond(
            ep,
            tau=float(hparams.get("tau", 20.0)),
            lam=float(hparams.get("lam", 0.5)),
            alpha=float(hparams.get("alpha", 1.0)),
        )
    elif method == "random":
        return predict_random(ep, rng_seed=42)
    elif method == "majority":
        return predict_majority(ep)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment(
    *,
    dataset: str,
    encoders: list[str],
    shots_list: list[int],
    imbalance_levels: list[float],
    n_episodes: int,
    split: str,
    hparams_path: Path,
    methods: list[str],
    csv_path: Path,
    feature_dir: Path = FEATURE_DIR,
    support_base_seed: int = 0,
    force: bool = False,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    planned_keys = {
        (dataset, enc, method, int(shots), float(imb), ep_id)
        for enc in encoders for method in methods for shots in shots_list
        for imb in imbalance_levels for ep_id in range(n_episodes)
    }
    prepare_csv(csv_path, planned_keys, force)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    # Summed confusion matrices over balanced episodes, keyed "{encoder}_{shots}shot_{method}"
    confusion: dict[str, np.ndarray] = {}

    try:
        for enc_key in encoders:
            logger.info("=== Encoder: %s ===", enc_key)
            pool_X, pool_y = load_features(dataset, enc_key, "pool", feature_dir)
            query_X, query_y = load_features(dataset, enc_key, split, feature_dir)
            n_classes = len(np.unique(pool_y))
            n_query_total = n_classes * 30

            for shots in shots_list:
                for imbalance in imbalance_levels:
                    logger.info("  shots=%d imbalance=%.2f", shots, imbalance)
                    t0 = time.time()

                    # Pre-sample all episodes once (paired across methods)
                    episodes = []
                    for ep_id in range(n_episodes):
                        ep = sample_episode(
                            pool_X=pool_X,
                            pool_y=pool_y,
                            query_X=query_X,
                            query_y=query_y,
                            n_classes=n_classes,
                            shots=shots,
                            n_query_total=n_query_total,
                            episode_id=ep_id,
                            support_seed=support_base_seed + ep_id,
                            imbalance=imbalance,
                        )
                        episodes.append(ep)

                    n_rep_eps = sum(ep.n_replaced_classes > 0 for ep in episodes)
                    if n_rep_eps:
                        logger.info(
                            "  query replace=True fallback: %d/%d episodes, %d class draws "
                            "(split=%s, imbalance=%.2f)",
                            n_rep_eps, len(episodes),
                            sum(ep.n_replaced_classes for ep in episodes), split, imbalance,
                        )

                    for method in methods:
                        hp = {}
                        if hparams_path.exists():
                            hp = load_hparams(hparams_path, enc_key, shots, method)

                        for ep_id, ep in enumerate(episodes):
                            preds = get_method_preds(method, ep, hp)
                            m = compute_metrics(ep.query_y, preds, n_classes=n_classes)
                            row = {
                                "dataset": dataset,
                                "encoder": enc_key,
                                "method": method,
                                "shots": shots,
                                "imbalance": imbalance,
                                "episode": ep_id,
                                "split": split,
                                "n_query": len(ep.query_y),
                                "acc": round(m["acc"], 6),
                                "macro_f1": round(m["macro_f1"], 6),
                                "bal_acc": round(m["bal_acc"], 6),
                            }
                            writer.writerow(row)
                            if imbalance == 0.0:
                                key = f"{enc_key}_{shots}shot_{method}"
                                cm = confusion.get(key, np.zeros((n_classes, n_classes), dtype=np.int64))
                                np.add.at(cm, (ep.query_y, preds), 1)
                                confusion[key] = cm

                    csv_file.flush()
                    elapsed = time.time() - t0
                    logger.info(
                        "  shots=%d imbalance=%.2f done in %.1fs", shots, imbalance, elapsed
                    )
    finally:
        csv_file.close()

    if confusion:
        cm_path = csv_path.parent / "confusion_matrices.npz"
        np.savez(cm_path, **confusion)
        logger.info("Saved %d balanced-episode confusion matrices to %s", len(confusion), cm_path)


def make_smoke_features(dataset: str, encoders: list[str], feature_dir: Path) -> None:
    """Write tiny synthetic feature files for smoke testing (never into features/)."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_classes = 8
    D = 64
    for enc_key in encoders:
        prefix = f"{dataset}_{enc_key}"
        for split_name, n_per_class in [("pool", 50), ("val", 30), ("test", 30)]:
            n = n_per_class * n_classes
            X = rng.standard_normal((n, D)).astype(np.float32)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)
            np.save(feature_dir / f"{prefix}_{split_name}_X.npy", X)
            np.save(feature_dir / f"{prefix}_{split_name}_Y.npy", Y)
    pool_idx = np.arange(50 * n_classes, dtype=np.int64)
    np.save(feature_dir / f"{dataset}_pool_idx.npy", pool_idx)


def make_smoke_hparams(
    dataset: str, encoders: list[str], shots_list: list[int], config_dir: Path
) -> Path:
    """Write a minimal best_hparams.json for smoke mode (never into configs/)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    hp: dict = {"dataset": dataset, "timestamp": "smoke", "smoke": True, "encoders": {}}
    for enc in encoders:
        hp["encoders"][enc] = {}
        for shots in shots_list:
            hp["encoders"][enc][str(shots)] = {
                "logreg": {"C_reg": 1.0},
                "tim": {"tau": 20.0, "lam": 0.5, "alpha": 1.0},
                "tim_cond": {"tau": 20.0, "lam": 0.5, "alpha": 1.0},
            }
    path = config_dir / "best_hparams.json"
    with open(path, "w") as f:
        json.dump(hp, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run few-shot experiments.")
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument("--encoders", nargs="+", default=["resnet50", "dinov2_s"])
    parser.add_argument("--shots", nargs="+", type=int, default=SHOTS_LIST)
    parser.add_argument("--imbalance", nargs="+", type=float, default=IMBALANCE_LEVELS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument(
        "--final", action="store_true",
        help="Run on test split. Requires configs/best_hparams.json to exist."
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Synthetic features, 2 episodes, CPU only. Finishes in seconds."
    )
    parser.add_argument(
        "--smoke-dir", default=str(SMOKE_DIR),
        help="Where smoke mode writes its features/, configs/ and results/."
    )
    parser.add_argument("--out", default=None, help="Override output CSV path.")
    parser.add_argument(
        "--force", action="store_true",
        help="Replace rows already present in the output CSV instead of refusing."
    )
    args = parser.parse_args()

    hparams_path = CONFIG_DIR / "best_hparams.json"

    if args.smoke:
        smoke_dir = Path(args.smoke_dir)
        logger.info("[smoke] Generating synthetic features in %s ...", smoke_dir)
        smoke_encoders = ["resnet50"]
        smoke_shots = [1, 5]
        smoke_imbalance = [0.0, 1.0]
        smoke_features = smoke_dir / "features"
        make_smoke_features(args.dataset, smoke_encoders, smoke_features)
        hparams_path = make_smoke_hparams(
            args.dataset, smoke_encoders, smoke_shots, smoke_dir / "configs"
        )
        csv_path = smoke_dir / "results" / "smoke_results.csv"
        if csv_path.exists():
            csv_path.unlink()
        run_experiment(
            dataset=args.dataset,
            encoders=smoke_encoders,
            shots_list=smoke_shots,
            imbalance_levels=smoke_imbalance,
            n_episodes=2,
            split="test",
            hparams_path=hparams_path,
            methods=METHODS,
            csv_path=csv_path,
            feature_dir=smoke_features,
        )
        logger.info("[smoke] PASSED. CSV: %s", csv_path)
        return

    if args.final:
        error = validate_final_hparams(hparams_path)
        if error:
            logger.error("%s Refusing to run final experiments.", error)
            sys.exit(1)
        split = "test"
        csv_path = Path(args.out) if args.out else RESULTS_DIR / "results.csv"
        logger.info("Final run on TEST split using frozen hparams from %s", hparams_path)
    else:
        split = "val"
        csv_path = Path(args.out) if args.out else RESULTS_DIR / "val_results.csv"
        logger.info("Running on VAL split (not final).")

    run_experiment(
        dataset=args.dataset,
        encoders=args.encoders,
        shots_list=args.shots,
        imbalance_levels=args.imbalance,
        n_episodes=args.episodes,
        split=split,
        hparams_path=hparams_path,
        methods=args.methods,
        csv_path=csv_path,
        support_base_seed=args.support_seed,
        force=args.force,
    )
    logger.info("Done. Results written to %s", csv_path)


if __name__ == "__main__":
    main()
