"""
make_plots.py — generate tables, figures, and results/summary.md from results.csv.

All numbers come exclusively from results/results.csv; never hardcoded.

Usage:
  python make_plots.py [--csv results/results.csv] [--out-dir results]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")


def ci95_series(s: pd.Series) -> float:
    """95% t-interval half-width for a pandas Series."""
    arr = s.dropna().values
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return float(se * t_crit)


def fmt(mean: float, ci: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


# ---------------------------------------------------------------------------
# Table A — encoder x method x shots
# ---------------------------------------------------------------------------

def make_table_a(df: pd.DataFrame) -> str:
    """Accuracy mean ± 95% CI per (encoder, method, shots), balanced queries."""
    bal = df[df["imbalance"] == 0.0]
    grouped = (
        bal.groupby(["encoder", "method", "shots"])["acc"]
        .agg(["mean", ci95_series])
        .rename(columns={"mean": "acc_mean", "ci95_series": "acc_ci95"})
        .reset_index()
    )
    lines = ["## Table A — Accuracy by Encoder × Method × Shots (balanced queries)\n"]
    lines.append("| encoder | method | shots=1 | shots=5 | shots=10 | shots=20 |")
    lines.append("|---|---|---|---|---|---|")
    for (enc, method), sub in grouped.groupby(["encoder", "method"]):
        cells = []
        for s in [1, 5, 10, 20]:
            row = sub[sub["shots"] == s]
            if row.empty:
                cells.append("—")
            else:
                cells.append(fmt(float(row["acc_mean"]), float(row["acc_ci95"])))
        lines.append(f"| {enc} | {method} | {' | '.join(cells)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table B — transductive gain (paired, Experiment B)
# ---------------------------------------------------------------------------

def make_table_b(df: pd.DataFrame) -> str:
    """Paired delta_acc (method - proto) per encoder/shots, balanced."""
    from scipy.stats import wilcoxon

    bal = df[df["imbalance"] == 0.0].copy()
    lines = ["## Table B — Transductive Gain vs Prototype (balanced queries)\n"]
    lines.append("| encoder | method | shots | Δacc (mean±CI) | win_rate | wilcoxon_p |")
    lines.append("|---|---|---|---|---|---|")

    for (enc, method) in bal.groupby(["encoder", "method"]).groups:
        if method in ("proto", "random", "majority"):
            continue
        for shots in [1, 5, 10, 20]:
            sub_m = bal[(bal["encoder"] == enc) & (bal["method"] == method) & (bal["shots"] == shots)]
            sub_p = bal[(bal["encoder"] == enc) & (bal["method"] == "proto") & (bal["shots"] == shots)]
            if sub_m.empty or sub_p.empty:
                continue
            # Align by episode
            m = sub_m.sort_values("episode")["acc"].values
            p = sub_p.sort_values("episode")["acc"].values
            if len(m) != len(p):
                continue
            diffs = m - p
            d_mean = float(np.nanmean(diffs))
            d_ci = ci95_series(pd.Series(diffs))
            win_rate = float((m > p).mean())
            try:
                if (diffs == 0).all():
                    pval = 1.0
                else:
                    _, pval = wilcoxon(diffs, alternative="two-sided")
            except Exception:
                pval = float("nan")
            lines.append(
                f"| {enc} | {method} | {shots} | {fmt(d_mean, d_ci)} | {win_rate:.2f} | {pval:.4f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table C — imbalance degradation (Experiment C)
# ---------------------------------------------------------------------------

def make_table_c(df: pd.DataFrame) -> str:
    """Accuracy by method and imbalance level (5-shot, DINOv2 as primary)."""
    lines = ["## Table C — Accuracy Under Imbalanced Queries (shots=5)\n"]
    lines.append("| encoder | method | balanced | α=1.0 | α=0.3 |")
    lines.append("|---|---|---|---|---|")
    sub = df[df["shots"] == 5]
    for (enc, method), g in sub.groupby(["encoder", "method"]):
        cells = []
        for imb in [0.0, 1.0, 0.3]:
            r = g[g["imbalance"] == imb]
            if r.empty:
                cells.append("—")
            else:
                cells.append(fmt(float(r["acc"].mean()), ci95_series(r["acc"])))
        lines.append(f"| {enc} | {method} | {' | '.join(cells)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_acc_vs_shots(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 1: accuracy and macro-F1 vs shots per encoder/method."""
    bal = df[df["imbalance"] == 0.0]
    encoders = sorted(bal["encoder"].unique())
    methods = [m for m in ["proto", "logreg", "tim", "tim_cond"] if m in bal["method"].values]
    shots_vals = sorted(bal["shots"].unique())

    colors = {"proto": "#2196F3", "logreg": "#4CAF50", "tim": "#FF5722", "tim_cond": "#9C27B0"}
    styles = {"proto": "-o", "logreg": "-s", "tim": "-^", "tim_cond": "--^"}

    fig, axes = plt.subplots(len(encoders), 2, figsize=(12, 4 * len(encoders)))
    if len(encoders) == 1:
        axes = axes[np.newaxis, :]

    for row_i, enc in enumerate(encoders):
        for col_i, metric in enumerate(["acc", "macro_f1"]):
            ax = axes[row_i, col_i]
            for method in methods:
                sub = bal[(bal["encoder"] == enc) & (bal["method"] == method)]
                means, cis = [], []
                for s in shots_vals:
                    vals = sub[sub["shots"] == s][metric].dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    cis.append(ci95_series(vals))
                means = np.array(means)
                cis = np.array(cis)
                ax.errorbar(
                    shots_vals, means, yerr=cis,
                    fmt=styles.get(method, "-o"),
                    color=colors.get(method, "gray"),
                    label=method, capsize=4, linewidth=1.5,
                )
            ax.set_xlabel("Shots (K)")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{enc} — {metric.replace('_', ' ').title()}")
            ax.legend(fontsize=8)
            ax.set_xticks(shots_vals)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy & Macro-F1 vs Shots", fontsize=14)
    fig.tight_layout()
    out = out_dir / "fig1_acc_vs_shots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_transductive_gain(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2: bar chart of transductive gain over prototype."""
    bal = df[df["imbalance"] == 0.0]
    methods = [m for m in ["logreg", "tim", "tim_cond"] if m in bal["method"].values]
    encoders = sorted(bal["encoder"].unique())
    shots_vals = sorted(bal["shots"].unique())

    colors = {"logreg": "#4CAF50", "tim": "#FF5722", "tim_cond": "#9C27B0"}

    fig, axes = plt.subplots(1, len(encoders), figsize=(5 * len(encoders), 5), squeeze=False)

    for col_i, enc in enumerate(encoders):
        ax = axes[0, col_i]
        proto_data = bal[(bal["encoder"] == enc) & (bal["method"] == "proto")]
        x = np.arange(len(shots_vals))
        width = 0.8 / len(methods)
        for m_i, method in enumerate(methods):
            method_data = bal[(bal["encoder"] == enc) & (bal["method"] == method)]
            gains = []
            for s in shots_vals:
                p_acc = proto_data[proto_data["shots"] == s]["acc"].mean()
                m_acc = method_data[method_data["shots"] == s]["acc"].mean()
                gains.append(m_acc - p_acc if not np.isnan(p_acc) else np.nan)
            offset = (m_i - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, gains, width=width * 0.9,
                   color=colors.get(method, "gray"), label=method, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Shots (K)")
        ax.set_ylabel("Δ Accuracy (vs proto)")
        ax.set_title(f"{enc} — Transductive Gain")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in shots_vals])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Transductive Gain over Prototype", fontsize=14)
    fig.tight_layout()
    out = out_dir / "fig2_transductive_gain.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_imbalance(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3: accuracy vs imbalance level per method (5-shot)."""
    sub5 = df[df["shots"] == 5]
    encoders = sorted(sub5["encoder"].unique())
    methods = [m for m in ["proto", "logreg", "tim", "tim_cond"] if m in sub5["method"].values]
    imb_levels = sorted(sub5["imbalance"].unique())

    colors = {"proto": "#2196F3", "logreg": "#4CAF50", "tim": "#FF5722", "tim_cond": "#9C27B0"}
    styles = {"proto": "-o", "logreg": "-s", "tim": "-^", "tim_cond": "--^"}

    fig, axes = plt.subplots(1, len(encoders), figsize=(5 * len(encoders), 5), squeeze=False)
    for col_i, enc in enumerate(encoders):
        ax = axes[0, col_i]
        for method in methods:
            sub = sub5[(sub5["encoder"] == enc) & (sub5["method"] == method)]
            means, cis = [], []
            for imb in imb_levels:
                vals = sub[sub["imbalance"] == imb]["acc"].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                cis.append(ci95_series(vals))
            ax.errorbar(
                imb_levels, means, yerr=cis,
                fmt=styles.get(method, "-o"),
                color=colors.get(method, "gray"),
                label=method, capsize=4, linewidth=1.5,
            )
        ax.set_xlabel("Dirichlet α (lower = more imbalanced)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{enc} — Accuracy vs Imbalance (5-shot)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy Under Imbalanced Query Sets", fontsize=14)
    fig.tight_layout()
    out = out_dir / "fig3_imbalance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_confusion_matrix(df: pd.DataFrame, raw_results_dir: Path, out_dir: Path) -> None:
    """Figure 4: confusion matrices from saved episode predictions (if available)."""
    # Confusion matrices require per-episode predictions, not just aggregated metrics.
    # If the detailed predictions file exists, plot; otherwise note it's not available.
    cm_path = raw_results_dir / "confusion_matrices.npz"
    if not cm_path.exists():
        logger.info("No confusion matrix data found at %s; skipping Figure 4.", cm_path)
        return

    data = np.load(cm_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ["proto", "tim"]):
        key = f"dinov2_s_5shot_{method}"
        if key not in data:
            ax.set_title(f"{method} — not available")
            continue
        cm = data[key]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"DINOv2 5-shot: {method}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out = out_dir / "fig4_confusion_matrices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> str:
    lines = [
        "# Results Summary\n",
        "_Generated automatically from results.csv by make_plots.py._\n",
        "_All numbers are conditional on the data and hyperparameters described in README.md._\n",
        "",
    ]
    lines.append(make_table_a(df))
    lines.append("\n")
    lines.append(make_table_b(df))
    lines.append("\n")
    lines.append(make_table_c(df))
    lines.append("\n")

    # Narrative findings (conditional on CSV)
    lines.append("## Narrative Findings (auto-generated)\n")
    bal = df[df["imbalance"] == 0.0]
    for enc in sorted(bal["encoder"].unique()):
        for shots in [5, 10]:
            for method in ["tim", "logreg"]:
                sub_m = bal[(bal["encoder"] == enc) & (bal["method"] == method) & (bal["shots"] == shots)]
                sub_p = bal[(bal["encoder"] == enc) & (bal["method"] == "proto") & (bal["shots"] == shots)]
                if sub_m.empty or sub_p.empty:
                    continue
                m_acc = sub_m.sort_values("episode")["acc"].values
                p_acc = sub_p.sort_values("episode")["acc"].values
                if len(m_acc) != len(p_acc):
                    continue
                diffs = m_acc - p_acc
                d_mean = float(np.nanmean(diffs))
                d_ci = ci95_series(pd.Series(diffs))
                direction = "improved" if d_mean > 0 else "degraded"
                lines.append(
                    f"- **{enc}, {shots}-shot**: `{method}` {direction} over `proto` by "
                    f"{abs(d_mean):.3f} ± {d_ci:.3f} acc pts "
                    f"({'p<0.05' if abs(d_mean) > d_ci else 'not significant'})."
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and summary from results CSV.")
    parser.add_argument("--csv", default="results/results.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        raise SystemExit(1)

    FIGURES_DIR_OUT = Path(args.figures_dir)
    FIGURES_DIR_OUT.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Figures
    if len(df) > 0:
        fig_acc_vs_shots(df, FIGURES_DIR_OUT)
        fig_transductive_gain(df, FIGURES_DIR_OUT)
        if df["imbalance"].nunique() > 1:
            fig_imbalance(df, FIGURES_DIR_OUT)
        fig_confusion_matrix(df, out_dir, FIGURES_DIR_OUT)

    # Tables + summary
    summary = make_summary(df)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
