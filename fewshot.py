"""
fewshot.py — core episodic few-shot library.

Sampler, methods (proto, logreg, TIM, tim_cond), metrics.
All features are assumed to be L2-normalised float32 numpy arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One sampled episode ready for evaluation."""
    support_X: np.ndarray   # (K*C, D)
    support_y: np.ndarray   # (K*C,)
    query_X: np.ndarray     # (n_query, D)
    query_y: np.ndarray     # (n_query,)
    n_classes: int
    pool_indices: np.ndarray   # which pool rows were used as support
    query_indices: np.ndarray  # which split rows were used as query
    n_replaced_classes: int = 0  # classes drawn with replace=True (too few images)


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------

def sample_episode(
    *,
    pool_X: np.ndarray,
    pool_y: np.ndarray,
    query_X: np.ndarray,
    query_y: np.ndarray,
    n_classes: int,
    shots: int,
    n_query_total: int,
    episode_id: int,
    support_seed: int,
    imbalance: float = 0.0,
) -> Episode:
    """Sample one episode.

    Support is drawn from pool_X/pool_y.
    Query is drawn from query_X/query_y.
    The query RNG is seeded from (episode_id, imbalance) independently of
    support seed and method, satisfying the paired-design requirement.

    Parameters
    ----------
    pool_X, pool_y   : pre-extracted train-pool features and labels.
    query_X, query_y : val or test split features and labels.
    n_classes        : C (number of classes 0..C-1 must all be present in pool).
    shots            : K shots per class.
    n_query_total    : total query size = C*30 fixed.
    episode_id       : used to seed the query RNG.
    support_seed     : used to seed the support RNG.
    imbalance        : Dirichlet concentration alpha (0 => balanced = 30 per class).
    """
    # --- Support sampling ---
    sup_rng = np.random.default_rng(support_seed)
    sup_idx_list: list[int] = []
    for c in range(n_classes):
        class_pool_idx = np.where(pool_y == c)[0]
        if len(class_pool_idx) < shots:
            logger.warning(
                "Class %d has only %d pool images, fewer than %d shots. "
                "Using all.", c, len(class_pool_idx), shots
            )
            chosen = class_pool_idx
        else:
            chosen = sup_rng.choice(class_pool_idx, shots, replace=False)
        sup_idx_list.extend(chosen.tolist())
    pool_indices = np.array(sup_idx_list, dtype=np.int64)
    support_X = pool_X[pool_indices]
    support_y = pool_y[pool_indices]

    # --- Query count per class (balanced or Dirichlet) ---
    # Seed derived ONLY from (episode_id, imbalance) independent of method/shots.
    imb_int = int(round(imbalance * 1000))  # encode float as int for seed
    qry_rng = np.random.default_rng([episode_id, imb_int])

    if imbalance <= 0.0:
        counts = np.full(n_classes, n_query_total // n_classes, dtype=np.int64)
    else:
        p = qry_rng.dirichlet(imbalance * np.ones(n_classes))
        counts = qry_rng.multinomial(n_query_total, p).astype(np.int64)

    # --- Query sampling per class ---
    qry_idx_list: list[int] = []
    n_replaced_classes = 0
    for c in range(n_classes):
        class_q_idx = np.where(query_y == c)[0]
        needed = int(counts[c])
        if needed == 0:
            continue
        if len(class_q_idx) < needed:
            logger.debug(
                "Class %d query: need %d but only %d available; using replace=True.",
                c, needed, len(class_q_idx),
            )
            chosen = qry_rng.choice(class_q_idx, needed, replace=True)
            n_replaced_classes += 1
        else:
            chosen = qry_rng.choice(class_q_idx, needed, replace=False)
        qry_idx_list.extend(chosen.tolist())

    query_indices = np.array(qry_idx_list, dtype=np.int64)
    qry_rng.shuffle(query_indices)
    ep_query_X = query_X[query_indices]
    ep_query_y = query_y[query_indices]

    return Episode(
        support_X=support_X,
        support_y=support_y,
        query_X=ep_query_X,
        query_y=ep_query_y,
        n_classes=n_classes,
        pool_indices=pool_indices,
        query_indices=query_indices,
        n_replaced_classes=n_replaced_classes,
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def predict_proto(ep: Episode) -> np.ndarray:
    """Nearest-prototype cosine classifier.

    Prototype = mean of support features re-normalised to unit norm.
    Returns predicted labels for query set.
    """
    C = ep.n_classes
    D = ep.support_X.shape[1]
    protos = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        mask = ep.support_y == c
        if mask.sum() == 0:
            continue
        mean_feat = ep.support_X[mask].mean(0)
        norm = np.linalg.norm(mean_feat)
        protos[c] = mean_feat / (norm + 1e-8)

    sims = ep.query_X @ protos.T  # (n_query, C)
    return sims.argmax(1).astype(np.int64)


def predict_logreg(ep: Episode, *, C_reg: float = 1.0) -> np.ndarray:
    """Multinomial logistic regression fit on support set only.

    sklearn LogisticRegression with L2 penalty.
    Returns predicted labels for query set.
    """
    clf = LogisticRegression(
        C=C_reg,
        max_iter=1000,
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(ep.support_X, ep.support_y)
    return clf.predict(ep.query_X).astype(np.int64)


def _tim_core(
    support_X: np.ndarray,
    support_y: np.ndarray,
    query_X: np.ndarray,
    n_classes: int,
    *,
    tau: float = 20.0,
    lam: float = 0.5,
    alpha: float = 1.0,
    steps: int = 50,
    lr: float = 1e-2,
    use_marginal: bool = True,
) -> np.ndarray:
    """TIM-style transductive inference (vectorised, pure PyTorch).

    Loss = CE(support) - lam * (H(Y) - alpha * H(Y|X))
    with H(Y|X) minimised and H(Y) maximised (when use_marginal=True).

    When use_marginal=False: loss = CE + lam*alpha*H(Y|X)  => tim_cond ablation.

    Returns integer predicted labels for query_X.
    """
    zs = torch.tensor(support_X, dtype=torch.float32)
    ys = torch.tensor(support_y, dtype=torch.long)
    zq = torch.tensor(query_X, dtype=torch.float32)

    C = n_classes
    proto_init = torch.zeros(C, zs.shape[1])
    for c in range(C):
        mask = ys == c
        if mask.sum() > 0:
            proto_init[c] = zs[mask].mean(0)
    P = proto_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)

    for _ in range(steps):
        Pn = F.normalize(P, dim=1)
        ce = F.cross_entropy(tau * zs @ Pn.T, ys)
        pq = (tau * zq @ Pn.T).softmax(1)
        cond = -(pq * (pq + 1e-8).log()).sum(1).mean()  # H(Y|X) minimise
        if use_marginal:
            pm = pq.mean(0)
            marg = -(pm * (pm + 1e-8).log()).sum()  # H(Y) maximise
            loss = ce - lam * (marg - alpha * cond)
        else:
            loss = ce - lam * (-alpha * cond)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Pn = F.normalize(P, dim=1)
        preds = (zq @ Pn.T).argmax(1).numpy().astype(np.int64)
    return preds


def predict_tim(
    ep: Episode,
    *,
    tau: float = 20.0,
    lam: float = 0.5,
    alpha: float = 1.0,
    steps: int = 50,
    lr: float = 1e-2,
) -> np.ndarray:
    """Full TIM: CE(support) - lam*(H(Y) - alpha*H(Y|X))."""
    return _tim_core(
        ep.support_X, ep.support_y, ep.query_X, ep.n_classes,
        tau=tau, lam=lam, alpha=alpha, steps=steps, lr=lr, use_marginal=True,
    )


def predict_tim_cond(
    ep: Episode,
    *,
    tau: float = 20.0,
    lam: float = 0.5,
    alpha: float = 1.0,
    steps: int = 50,
    lr: float = 1e-2,
) -> np.ndarray:
    """TIM ablation: conditional entropy term only (no marginal-entropy term)."""
    return _tim_core(
        ep.support_X, ep.support_y, ep.query_X, ep.n_classes,
        tau=tau, lam=lam, alpha=alpha, steps=steps, lr=lr, use_marginal=False,
    )


# ---------------------------------------------------------------------------
# Sanity baselines
# ---------------------------------------------------------------------------

def predict_random(ep: Episode, *, rng_seed: int = 42) -> np.ndarray:
    """Predict uniformly at random from 0..C-1."""
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, ep.n_classes, size=len(ep.query_y)).astype(np.int64)


def predict_majority(ep: Episode) -> np.ndarray:
    """Predict the most frequent class of the query set (oracle sanity baseline).

    Uses query labels by design: it is a reference row, not a method.
    Ties go to the smallest class index.
    """
    classes, counts = np.unique(ep.query_y, return_counts=True)
    maj_class = int(classes[counts.argmax()])
    return np.full(len(ep.query_y), maj_class, dtype=np.int64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, float]:
    """Compute accuracy, macro-F1, balanced accuracy.

    Macro-F1 and balanced accuracy are computed only over classes present
    in y_true (some classes may be absent under strong imbalance).
    This is stated in the README.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n == 0:
        return {"acc": float("nan"), "macro_f1": float("nan"), "bal_acc": float("nan")}

    acc = float((y_true == y_pred).mean())

    present_classes = np.unique(y_true)
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []

    for c in present_classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else float("nan")
    bal_acc = float(np.mean(per_class_recall)) if per_class_recall else float("nan")

    return {"acc": acc, "macro_f1": macro_f1, "bal_acc": bal_acc}


def ci95(values: np.ndarray) -> float:
    """Two-sided 95% t-interval half-width."""
    from scipy import stats
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return float(se * t_crit)


def aggregate_episodes(rows: list[dict]) -> dict[str, float]:
    """Mean and 95% CI for acc, macro_f1, bal_acc over episode rows."""
    result: dict[str, float] = {}
    for key in ("acc", "macro_f1", "bal_acc"):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        result[f"{key}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
        result[f"{key}_ci95"] = ci95(vals)
    return result


# ---------------------------------------------------------------------------
# Improvement / paired metrics (Experiment B/C)
# ---------------------------------------------------------------------------

def paired_improvement(
    method_rows: list[dict],
    baseline_rows: list[dict],
) -> dict[str, float]:
    """Paired improvement of method over baseline (proto or logreg).

    Computes per-episode differences then aggregates.
    Returns delta_acc, delta_macro_f1, delta_bal_acc (mean +/- CI),
    relative error reduction, win rate, and Wilcoxon p-value.
    """
    from scipy.stats import wilcoxon

    assert len(method_rows) == len(baseline_rows), "Must be paired episodes"

    result: dict[str, float] = {}
    for key in ("acc", "macro_f1", "bal_acc"):
        m_vals = np.array([r[key] for r in method_rows], dtype=np.float64)
        b_vals = np.array([r[key] for r in baseline_rows], dtype=np.float64)
        diffs = m_vals - b_vals
        valid = diffs[~np.isnan(diffs)]
        result[f"delta_{key}_mean"] = float(np.nanmean(diffs))
        result[f"delta_{key}_ci95"] = ci95(valid)

    b_acc = np.array([r["acc"] for r in baseline_rows], dtype=np.float64)
    m_acc = np.array([r["acc"] for r in method_rows], dtype=np.float64)
    err_b = 1.0 - b_acc
    rer = np.where(err_b > 1e-9, (err_b - (1.0 - m_acc)) / err_b, 0.0)
    result["rel_err_reduction_mean"] = float(np.nanmean(rer))
    result["win_rate"] = float((m_acc > b_acc).mean())

    diffs_acc = (m_acc - b_acc)
    diffs_acc = diffs_acc[~np.isnan(diffs_acc)]
    if len(diffs_acc) >= 10 and not (diffs_acc == 0).all():
        try:
            _, pval = wilcoxon(diffs_acc, alternative="two-sided")
            result["wilcoxon_p"] = float(pval)
        except Exception:
            result["wilcoxon_p"] = float("nan")
    else:
        result["wilcoxon_p"] = float("nan")

    return result
