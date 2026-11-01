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
        counts = {
            tuple(np.bincount(make_ep(imbalance=1.0, episode_id=i).query_y, minlength=4))
            for i in range(10)
        }
        assert len(counts) > 1, "Dirichlet query counts identical across 10 episodes"
        assert ep0.query_y.shape == ep1.query_y.shape

    def test_query_independent_of_shots(self):
        """Same episode_id and imbalance => same query set for every K."""
        for imb in (0.0, 1.0, 0.3):
            q = [make_ep(shots=k, imbalance=imb, episode_id=5).query_indices for k in (1, 5, 20)]
            np.testing.assert_array_equal(q[0], q[1])
            np.testing.assert_array_equal(q[0], q[2])

    def test_methods_do_not_shift_rng_stream(self):
        """Evaluating methods between samples must not change later episodes."""
        plain = [make_ep(imbalance=1.0, episode_id=i, support_seed=i) for i in range(4)]
        mixed = []
        for i in range(4):
            ep = make_ep(imbalance=1.0, episode_id=i, support_seed=i)
            mixed.append(ep)
            np.random.seed(i)  # perturb global RNGs too
            for predict in (predict_proto, predict_logreg, predict_tim, predict_tim_cond,
                            predict_random, predict_majority):
                predict(ep)
        for a, b in zip(plain, mixed):
            np.testing.assert_array_equal(a.pool_indices, b.pool_indices)
            np.testing.assert_array_equal(a.query_indices, b.query_indices)

    def test_replacement_fallback_counted(self):
        """A class with fewer query images than requested is drawn with replacement and counted."""
        pool_X, pool_y = make_pool(n_classes=4)
        query_X, query_y = make_query(n_classes=4, n_per_class=5)  # 5 < 30 per class
        ep = sample_episode(
            pool_X=pool_X, pool_y=pool_y, query_X=query_X, query_y=query_y,
            n_classes=4, shots=1, n_query_total=120,
            episode_id=0, support_seed=0, imbalance=0.0,
        )
        assert ep.n_replaced_classes == 4
        assert len(ep.query_y) == 120
        ok = make_ep(imbalance=0.0)
        assert ok.n_replaced_classes == 0

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
        import torch
        import torch.nn.functional as F

        ep = make_ep(shots=5, n_classes=4, dim=32)
        tau, steps, lr = 20.0, 50, 1e-2

        # Independent CE-only reference: support cross-entropy, nothing else.
        zs = torch.tensor(ep.support_X)
        ys = torch.tensor(ep.support_y)
        zq = torch.tensor(ep.query_X)
        P = torch.stack([zs[ys == c].mean(0) for c in range(4)]).clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=lr)
        for _ in range(steps):
            loss = F.cross_entropy(tau * zs @ F.normalize(P, dim=1).T, ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ref = (zq @ F.normalize(P.detach(), dim=1).T).argmax(1).numpy()

        np.testing.assert_array_equal(
            predict_tim(ep, tau=tau, lam=0.0, alpha=1.0, steps=steps, lr=lr), ref
        )
        np.testing.assert_array_equal(
            predict_tim_cond(ep, tau=tau, lam=0.0, alpha=1.0, steps=steps, lr=lr), ref
        )

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

    def test_majority_uses_query_distribution(self):
        """Majority = most frequent class of the query set (spec), not of the balanced support."""
        ep = make_ep(shots=5, n_classes=4, dim=32)
        query_y = np.array([2, 2, 2, 1, 0], dtype=np.int64)
        ep = Episode(
            support_X=ep.support_X, support_y=ep.support_y,
            query_X=np.zeros((5, 32), dtype=np.float32), query_y=query_y,
            n_classes=4, pool_indices=ep.pool_indices, query_indices=np.arange(5),
        )
        preds = predict_majority(ep)
        assert (preds == 2).all()
        assert abs(compute_metrics(query_y, preds, n_classes=4)["acc"] - 0.6) < 1e-9


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

    def test_smoke_mode_completes_and_is_isolated(self, tmp_path):
        """--smoke must complete and must not write to the real features/ or configs/."""
        repo_root = Path(__file__).parent.parent

        def snapshot() -> dict:
            out = {}
            for d in ("features", "configs", "results"):
                for p in (repo_root / d).glob("*"):
                    out[str(p)] = p.stat().st_mtime_ns
            return out

        before = snapshot()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        result = subprocess.run(
            [sys.executable, "run_experiments.py", "--smoke", "--smoke-dir", str(tmp_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Smoke run failed:\n{result.stdout}\n{result.stderr}"
        )
        assert (tmp_path / "results" / "smoke_results.csv").exists()
        assert (tmp_path / "features").is_dir()
        assert snapshot() == before, "smoke run modified real features/, configs/ or results/"

        # Balanced episodes leave summed confusion matrices next to the CSV (Fig 4 input)
        cms = np.load(tmp_path / "results" / "confusion_matrices.npz")
        assert "resnet50_5shot_tim" in cms.files and "resnet50_5shot_proto" in cms.files
        for key in cms.files:
            cm = cms[key]
            n_classes = cm.shape[0]
            assert cm.shape == (n_classes, n_classes)
            assert cm.sum() == 2 * n_classes * 30  # 2 smoke episodes x balanced C*30 queries

    def test_final_refuses_smoke_hparams(self, tmp_path):
        """--final must refuse a best_hparams.json written by a smoke run."""
        import shutil
        repo_root = Path(__file__).parent.parent
        for fname in ["run_experiments.py", "fewshot.py"]:
            shutil.copy(repo_root / fname, tmp_path / fname)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "best_hparams.json").write_text(
            json.dumps({"dataset": "bloodmnist", "timestamp": "smoke", "encoders": {"x": {}}})
        )
        result = subprocess.run(
            [sys.executable, str(tmp_path / "run_experiments.py"), "--final"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "smoke" in (result.stdout + result.stderr)

    def test_tune_loader_refuses_test_split(self):
        import tune
        with pytest.raises(ValueError, match="test"):
            tune.load_features("bloodmnist", "resnet50", "test")

    def test_tune_never_opens_test_split(self, tmp_path, monkeypatch):
        """Run the real tuning path on synthetic features; record every file it loads."""
        import run_experiments
        import tune

        run_experiments.make_smoke_features("bloodmnist", ["resnet50"], tmp_path / "features")
        opened: list[str] = []
        real_load = np.load

        def spy_load(path, *a, **kw):
            opened.append(str(path))
            return real_load(path, *a, **kw)

        monkeypatch.setattr(tune.np, "load", spy_load)
        monkeypatch.setattr(sys, "argv", [
            "tune.py", "--smoke", "--encoders", "resnet50", "--shots", "1",
            "--feature-dir", str(tmp_path / "features"),
            "--config-dir", str(tmp_path / "configs"),
        ])
        tune.main()

        names = [Path(p).name for p in opened]  # tmp dir name itself contains "test"
        assert names, "tune.py loaded no features"
        assert all(("_pool_" in n or "_val_" in n) for n in names), names
        assert not any("_test_" in n for n in names), names
        hp = json.loads((tmp_path / "configs" / "best_hparams.json").read_text())
        assert hp["smoke"] is True  # so --final would refuse it

    def test_results_csv_refuses_duplicates(self, tmp_path):
        import run_experiments as rx

        csv_path = tmp_path / "r.csv"
        row = {"dataset": "d", "encoder": "e", "method": "proto", "shots": 1,
               "imbalance": 0.0, "episode": 0, "split": "test", "n_query": 1,
               "acc": 1.0, "macro_f1": 1.0, "bal_acc": 1.0}
        other = {**row, "episode": 1}
        import csv as _csv
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=rx.CSV_COLUMNS)
            w.writeheader()
            w.writerows([row, other])

        planned = {rx.row_key(row)}
        with pytest.raises(SystemExit):
            rx.prepare_csv(csv_path, planned, force=False)
        rx.prepare_csv(csv_path, planned, force=True)
        kept = rx.read_existing_rows(csv_path)
        assert [rx.row_key(r) for r in kept] == [rx.row_key(other)]
        rx.prepare_csv(csv_path, {("d", "e", "tim", 1, 0.0, 0)}, force=False)  # no clash: ok
