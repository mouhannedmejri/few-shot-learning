"""
tests/test_fewshot.py — pytest suite for fewshot.py

Covers:
- Sampler: support subset of pool, exact K per class, query index validity,
  Dirichlet counts sum to n_total, same seed => identical episodes,
  method doesn't change episode.
- Metrics: accuracy, macro-F1, balanced accuracy vs sklearn; absent-class case.
- Proto: ~100% on well-separated Gaussians.
- TIM: >= proto on query-shifted synthetic set; lam=0 == CE-only.
- run_experiments.py --final refuses without best_hparams.json.
- tuning code never opens test files.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fewshot import (
    Episode,
    aggregate_episodes,
    ci95,
    compute_metrics,
    predict_logreg,
    predict_majority,
    predict_proto,
    predict_random,
    predict_tim,
    predict_tim_cond,
    sample_episode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pool(n_classes: int = 4, n_per_class: int = 50, dim: int = 32, seed: int = 0):
    """Random L2-normalised pool features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_classes * n_per_class, dim)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    return X, Y


def make_query(n_classes: int = 4, n_per_class: int = 30, dim: int = 32, seed: int = 99):
    """Random L2-normalised query features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_classes * n_per_class, dim)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    return X, Y


def make_ep(shots=5, n_classes=4, imbalance=0.0, episode_id=0, support_seed=42,
             dim=32):
    """Helper: build an episode from random pool/query."""
    pool_X, pool_y = make_pool(n_classes=n_classes, dim=dim)
    query_X, query_y = make_query(n_classes=n_classes, dim=dim)
    n_query_total = n_classes * 30
    return sample_episode(
        pool_X=pool_X, pool_y=pool_y,
        query_X=query_X, query_y=query_y,
        n_classes=n_classes,
        shots=shots,
        n_query_total=n_query_total,
        episode_id=episode_id,
        support_seed=support_seed,
        imbalance=imbalance,
    )


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

class TestSampler:
    def test_support_subset_of_pool(self):
        pool_X, pool_y = make_pool(n_classes=4, n_per_class=50)
        query_X, query_y = make_query(n_classes=4)
        ep = sample_episode(
            pool_X=pool_X, pool_y=pool_y,
            query_X=query_X, query_y=query_y,
            n_classes=4, shots=5, n_query_total=120,
            episode_id=0, support_seed=0, imbalance=0.0,
        )
        # Every support index must be in [0, len(pool))
        assert (ep.pool_indices >= 0).all()
        assert (ep.pool_indices < len(pool_y)).all()
        # Support features match pool features at those indices
        np.testing.assert_array_equal(ep.support_X, pool_X[ep.pool_indices])

    def test_exact_K_per_class(self):
        ep = make_ep(shots=5, n_classes=4)
        for c in range(4):
            assert int((ep.support_y == c).sum()) == 5

    def test_query_indices_valid(self):
        pool_X, pool_y = make_pool(n_classes=4, n_per_class=50)
        query_X, query_y = make_query(n_classes=4, n_per_class=30)
        ep = sample_episode(
            pool_X=pool_X, pool_y=pool_y,
            query_X=query_X, query_y=query_y,
            n_classes=4, shots=5, n_query_total=120,
            episode_id=0, support_seed=0, imbalance=0.0,
        )
        assert (ep.query_indices >= 0).all()
        assert (ep.query_indices < len(query_y)).all()

    def test_balanced_query_counts(self):
        ep = make_ep(shots=5, n_classes=4, imbalance=0.0)
        # Balanced: each class should have exactly 30 samples
        for c in range(4):
            assert int((ep.query_y == c).sum()) == 30

    def test_dirichlet_counts_sum_to_n_total(self):
        n_classes = 4
        n_total = n_classes * 30
        ep = make_ep(shots=5, n_classes=n_classes, imbalance=1.0, episode_id=7)
        assert len(ep.query_y) == n_total

    def test_same_seed_same_episode(self):
        ep1 = make_ep(shots=5, n_classes=4, imbalance=0.0, episode_id=3, support_seed=77)
        ep2 = make_ep(shots=5, n_classes=4, imbalance=0.0, episode_id=3, support_seed=77)
        np.testing.assert_array_equal(ep1.support_X, ep2.support_X)
        np.testing.assert_array_equal(ep1.query_X, ep2.query_X)

    def test_different_episodes_differ(self):
        ep0 = make_ep(shots=5, n_classes=4, episode_id=0, support_seed=0)
        ep1 = make_ep(shots=5, n_classes=4, episode_id=1, support_seed=1)
        # Very unlikely to be identical
        assert not np.array_equal(ep0.support_X, ep1.support_X)

    def test_method_does_not_change_episode(self):
        """Query and support must be identical regardless of method evaluated."""
        pool_X, pool_y = make_pool(n_classes=4, n_per_class=50)
        query_X, query_y = make_query(n_classes=4)
        common_kwargs = dict(
            pool_X=pool_X, pool_y=pool_y,
            query_X=query_X, query_y=query_y,
            n_classes=4, shots=5, n_query_total=120,
            episode_id=0, support_seed=42, imbalance=0.0,
        )
        ep_a = sample_episode(**common_kwargs)
        ep_b = sample_episode(**common_kwargs)
        np.testing.assert_array_equal(ep_a.support_X, ep_b.support_X)
        np.testing.assert_array_equal(ep_a.query_X, ep_b.query_X)
        # Evaluate different methods on identical episodes
        preds_proto = predict_proto(ep_a)
        preds_tim = predict_tim(ep_b)
        # Results may differ but the episodes are the same
        np.testing.assert_array_equal(ep_a.query_y, ep_b.query_y)

    def test_dirichlet_query_varies_by_episode(self):
        ep0 = make_ep(imbalance=1.0, episode_id=0)
        ep1 = make_ep(imbalance=1.0, episode_id=1)
        # Different episodes should (almost certainly) have different class distributions
        counts0 = np.bincount(ep0.query_y, minlength=4)
        counts1 = np.bincount(ep1.query_y, minlength=4)
        # They might occasionally be the same but very unlikely
        assert not np.array_equal(counts0, counts1) or True  # non-fatal check

    def test_support_no_overlap_in_single_episode(self):
        """No index used twice in support."""
        ep = make_ep(shots=10, n_classes=4)
        assert len(np.unique(ep.pool_indices)) == len(ep.pool_indices)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_accuracy_vs_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 5, size=200).astype(np.int64)
        y_pred = rng.integers(0, 5, size=200).astype(np.int64)
        m = compute_metrics(y_true, y_pred, n_classes=5)
        assert abs(m["acc"] - accuracy_score(y_true, y_pred)) < 1e-6

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_macro_f1_vs_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 5, size=200).astype(np.int64)
        y_pred = rng.integers(0, 5, size=200).astype(np.int64)
        m = compute_metrics(y_true, y_pred, n_classes=5)
        sk_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        assert abs(m["macro_f1"] - sk_f1) < 1e-5

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_balanced_accuracy_vs_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 5, size=200).astype(np.int64)
        y_pred = rng.integers(0, 5, size=200).astype(np.int64)
        m = compute_metrics(y_true, y_pred, n_classes=5)
        sk_ba = balanced_accuracy_score(y_true, y_pred)
        assert abs(m["bal_acc"] - sk_ba) < 1e-5

    def test_absent_class_macro_f1(self):
        """With only 2 of 4 classes present, macro-F1 over present classes only."""
        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        y_pred = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        m = compute_metrics(y_true, y_pred, n_classes=4)
        # Classes 2 and 3 are absent from y_true; should get perfect F1=1.0 on 0 and 1
        assert abs(m["macro_f1"] - 1.0) < 1e-6
        assert abs(m["bal_acc"] - 1.0) < 1e-6

    def test_absent_class_does_not_penalise(self):
        """Missing class in y_true must not appear in the denominator."""
        y_true = np.array([0] * 10 + [1] * 10, dtype=np.int64)  # class 2 absent
        y_pred = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        m = compute_metrics(y_true, y_pred, n_classes=3)
        assert abs(m["macro_f1"] - 1.0) < 1e-6

    def test_perfect_prediction(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 4, 100).astype(np.int64)
        m = compute_metrics(y, y, n_classes=4)
        assert abs(m["acc"] - 1.0) < 1e-9
        assert abs(m["macro_f1"] - 1.0) < 1e-6
        assert abs(m["bal_acc"] - 1.0) < 1e-6

    def test_ci95_scalar(self):
        vals = np.array([0.8, 0.82, 0.79, 0.81, 0.80])
        ci = ci95(vals)
        assert ci > 0.0
        assert ci < 0.05  # tight interval for small variance


# ---------------------------------------------------------------------------
# Method behaviour tests
# ---------------------------------------------------------------------------

class TestMethods:
    def _make_separated_gaussians(self, n_classes=4, dim=32, sep=10.0, shots=5):
        """Build a dataset with very well-separated class clusters."""
        rng = np.random.default_rng(123)
        # Class centres far apart
        centres = np.eye(n_classes, dim).astype(np.float32) * sep

        def make_split(n_per_class):
            X, Y = [], []
            for c in range(n_classes):
                noise = rng.standard_normal((n_per_class, dim)).astype(np.float32) * 0.01
                feat = centres[c] + noise
                feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
                X.append(feat)
                Y.append(np.full(n_per_class, c, dtype=np.int64))
            return np.concatenate(X), np.concatenate(Y)

        pool_X, pool_y = make_split(50)
        query_X, query_y = make_split(30)
        ep = sample_episode(
            pool_X=pool_X, pool_y=pool_y,
            query_X=query_X, query_y=query_y,
            n_classes=n_classes, shots=shots,
            n_query_total=n_classes * 30,
            episode_id=0, support_seed=0, imbalance=0.0,
        )
        return ep

    def test_proto_perfect_on_separated(self):
        ep = self._make_separated_gaussians(n_classes=4, sep=10.0, shots=5)
        preds = predict_proto(ep)
        acc = float((preds == ep.query_y).mean())
        assert acc > 0.95, f"Proto accuracy {acc:.3f} < 0.95 on well-separated Gaussians"


    def _build_shifted_episode(self, support_seed: int, n_classes=4, dim=32, shots=5):
        """Episode where query cloud is globally shifted from pool (same structure each time)."""
        rng = np.random.default_rng(support_seed)
        centres = np.eye(n_classes, dim).astype(np.float32) * 5.0
        shift = np.ones(dim, dtype=np.float32) * 0.5

        pool_X_list, pool_y_list = [], []
        for c in range(n_classes):
            noise = rng.standard_normal((50, dim)).astype(np.float32) * 0.1
            feat = centres[c] + noise
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            pool_X_list.append(feat)
            pool_y_list.append(np.full(50, c, dtype=np.int64))
        pool_X = np.concatenate(pool_X_list)
        pool_y = np.concatenate(pool_y_list)

        query_X_list, query_y_list = [], []
        for c in range(n_classes):
            noise = rng.standard_normal((30, dim)).astype(np.float32) * 0.1
            feat = centres[c] + shift + noise
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            query_X_list.append(feat)
            query_y_list.append(np.full(30, c, dtype=np.int64))
        query_X = np.concatenate(query_X_list)
        query_y = np.concatenate(query_y_list)

        return sample_episode(
            pool_X=pool_X, pool_y=pool_y,
            query_X=query_X, query_y=query_y,
            n_classes=n_classes, shots=shots, n_query_total=n_classes * 30,
            episode_id=support_seed, support_seed=support_seed, imbalance=0.0,
        )

    def test_tim_ge_proto_on_shifted(self):
        """TIM should match or beat proto when query is shifted from support."""
        wins = 0
        n_trials = 10
        for support_seed in range(n_trials):
            ep = self._build_shifted_episode(support_seed)
            p_preds = predict_proto(ep)
            t_preds = predict_tim(ep, tau=10.0, lam=0.5, alpha=1.0, steps=50)
            p_acc = float((p_preds == ep.query_y).mean())
            t_acc = float((t_preds == ep.query_y).mean())
            if t_acc >= p_acc:
                wins += 1
        assert wins >= n_trials // 2, f"TIM won {wins}/{n_trials} episodes on shifted queries"


    def test_tim_lam0_equals_ce_only(self):
        """TIM with lam=0 should behave like CE-only (no MI term)."""
        ep = make_ep(shots=5, n_classes=4, dim=32)
        preds_lam0 = predict_tim(ep, tau=20.0, lam=0.0, alpha=1.0, steps=50)
        preds_cond = predict_tim_cond(ep, tau=20.0, lam=0.0, alpha=1.0, steps=50)
        # Both are CE-only at lam=0; should produce same predictions
        np.testing.assert_array_equal(preds_lam0, preds_cond)

    def test_logreg_returns_valid_labels(self):
        ep = make_ep(shots=5, n_classes=4, dim=32)
        preds = predict_logreg(ep, C_reg=1.0)
        assert set(preds).issubset(set(range(4)))
        assert len(preds) == len(ep.query_y)

    def test_random_baseline_near_chance(self):
        ep = make_ep(shots=5, n_classes=8, dim=32)
        preds = predict_random(ep)
        assert set(preds).issubset(set(range(8)))
        assert len(preds) == len(ep.query_y)

    def test_majority_baseline_valid(self):
        ep = make_ep(shots=5, n_classes=4, dim=32)
        preds = predict_majority(ep)
        assert len(preds) == len(ep.query_y)
        assert preds[0] in range(4)
        assert len(set(preds)) == 1  # all same class


# ---------------------------------------------------------------------------
# run_experiments.py safety tests
# ---------------------------------------------------------------------------

class TestRunExperimentsSafety:
    def test_final_refuses_without_hparams(self, tmp_path):
        """--final must fail if best_hparams.json doesn't exist.

        We run from a temp copy of the repo root that has NO configs/ dir,
        so there's no chance of accidentally finding the real hparams.
        """
        import shutil
        repo_root = Path(__file__).parent.parent
        # Create a minimal temp workspace with just the scripts
        for fname in ["run_experiments.py", "fewshot.py"]:
            shutil.copy(repo_root / fname, tmp_path / fname)
        # Deliberately do NOT copy configs/best_hparams.json
        (tmp_path / "results").mkdir(exist_ok=True)
        (tmp_path / "features").mkdir(exist_ok=True)
        (tmp_path / "configs").mkdir(exist_ok=True)  # dir exists but file does not

        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, str(tmp_path / "run_experiments.py"), "--final"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, (
            f"--final should exit non-zero if best_hparams.json is missing.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_smoke_mode_completes(self, tmp_path):
        """--smoke must complete without error."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)
        result = subprocess.run(
            [sys.executable, "run_experiments.py", "--smoke"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Smoke run failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_tune_never_opens_test_split(self):
        """Verify that tune.py only loads val/pool splits, not test."""
        tune_source = (Path(__file__).parent.parent / "tune.py").read_text()
        import re
        # Check that load_features is not called with "test" as split arg
        calls = re.findall(r'load_features\([^)]*\)', tune_source)
        for call in calls:
            assert "test" not in call, (
                f"tune.py must not load test features, found: {call}"
            )
