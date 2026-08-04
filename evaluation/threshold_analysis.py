"""
Figures for the threshold tuning.

The main evaluation figures compare the three models against each other. These
ones are about the cutoff instead: what changes when you stop using 0.5 and
start using the thresholds picked in final_model/build_final_pipeline.py.

Writes into evaluation/figures/:
  threshold_sweep.png             precision/recall/F1 across every cutoff
  confusion_by_threshold.png      confusion matrix at each operating point
  pr_curve_operating_points.png   where each cutoff sits on the PR curve
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
)

# -----------------------------
# Paths
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODEL_PATH = os.path.join(ROOT, "final_model", "surge_prediction_pipeline.pkl")
THRESHOLD_PATH = os.path.join(ROOT, "final_model", "pipeline_thresholds.json")
FEATURE_FILE = os.path.join(ROOT, "feature_matrix_era2022.csv")
TARGET_FILE = os.path.join(ROOT, "targets_era2022.csv")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET = "y_surge_next_week"

PRECISION_C = "#2a78d6"
RECALL_C = "#eb6834"
F1_C = "#1baf7a"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e2dd"

# Labels for the operating points, in the order we want them shown.
POINTS = [
    ("default_0.5", "Default 0.5", "#8a8a86"),
    ("high_precision", "High precision", PRECISION_C),
    ("balanced", "Balanced", F1_C),
    ("high_recall", "High recall", RECALL_C),
]


def style(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def load():
    features = pd.read_csv(FEATURE_FILE)
    targets = pd.read_csv(TARGET_FILE)
    df = features.merge(targets, on=["state_territory", "week_end"], how="inner")
    test = df[df["split_era"] == "test"]

    pipe = joblib.load(MODEL_PATH)
    proba = pipe.predict_proba(test.drop(columns=[TARGET]))[:, 1]

    thresholds = json.load(open(THRESHOLD_PATH))["thresholds"]
    thresholds = {"default_0.5": 0.5, **thresholds}
    return test[TARGET].to_numpy(), proba, thresholds


def plot_sweep(y, proba, thresholds):
    """Precision, recall and F1 at every possible cutoff, with ours marked."""
    grid = np.linspace(0.01, 0.99, 197)
    prec, rec, f1, n_flagged = [], [], [], []
    for t in grid:
        pred = (proba >= t).astype(int)
        prec.append(precision_score(y, pred, zero_division=0))
        rec.append(recall_score(y, pred, zero_division=0))
        f1.append(f1_score(y, pred, zero_division=0))
        n_flagged.append(int(pred.sum()))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    style(ax)

    # Past a certain cutoff the model flags almost nothing, so precision jumps
    # around on a handful of predictions. Grey that part out so the spike near
    # 0.7 isn't read as the model suddenly getting good.
    n_flagged = np.array(n_flagged)
    sparse = grid[n_flagged < 10]
    if len(sparse):
        ax.axvspan(sparse.min(), 1.0, color=GRID, alpha=0.55, zorder=0)
        ax.annotate("fewer than 10 weeks flagged,\nscores unstable here",
                    xy=(min(sparse.min() + 0.015, 0.97), 0.30),
                    fontsize=8, color=INK_SOFT, ha="left")

    ax.plot(grid, rec, color=RECALL_C, linewidth=2, label="Recall")
    ax.plot(grid, prec, color=PRECISION_C, linewidth=2, label="Precision")
    ax.plot(grid, f1, color=F1_C, linewidth=2, label="F1")

    for key, label, color in POINTS:
        t = thresholds[key]
        ax.axvline(t, color=INK_SOFT, linewidth=1, linestyle="--", alpha=0.65)
        ax.annotate(
            f"{label}\n{t:.3f}",
            xy=(t, 1.0), xytext=(t, 1.045),
            ha="center", va="bottom", fontsize=8, color=INK_SOFT,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Decision threshold", fontsize=11, color=INK)
    ax.set_ylabel("Score", fontsize=11, color=INK)
    ax.set_title("What the decision threshold costs and buys",
                 fontsize=12, color=INK, fontweight="bold", pad=26)
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "threshold_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_confusions(y, proba, thresholds):
    fig, axes = plt.subplots(1, len(POINTS), figsize=(4.0 * len(POINTS), 4.1))

    for ax, (key, label, _) in zip(axes, POINTS):
        t = thresholds[key]
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        caught = cm[1, 1]

        ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(f"{label}  (thr {t:.3f})\n{caught} of {cm[1].sum()} surges caught",
                     fontsize=10.5, color=INK, pad=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No surge", "Surge"], fontsize=9)
        ax.set_yticklabels(["No surge", "Surge"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9.5, color=INK_SOFT)
        ax.set_ylabel("Actual", fontsize=9.5, color=INK_SOFT)

        thresh = cm.max() / 2.0
        names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}",
                        ha="center", va="center", fontsize=10.5,
                        color="white" if cm[i, j] > thresh else INK,
                        fontweight="bold")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Same model, four different cutoffs (2022+ test set)",
                 fontsize=12.5, color=INK, y=1.03, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "confusion_by_threshold.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_pr_points(y, proba, thresholds):
    prec, rec, thr = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    style(ax)
    ax.plot(rec, prec, color=INK_SOFT, linewidth=2, label=f"XGBoost (PR-AUC {ap:.3f})")
    ax.axhline(y.mean(), color=INK_SOFT, linewidth=1.2, linestyle="--",
               label=f"No-skill floor ({y.mean():.3f})")

    for key, label, color in POINTS:
        t = thresholds[key]
        pred = (proba >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        ax.scatter([r], [p], s=90, color=color, zorder=5,
                   edgecolor="white", linewidth=1.6, label=f"{label} ({t:.3f})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=11, color=INK)
    ax.set_ylabel("Precision", fontsize=11, color=INK)
    ax.set_title("Where each cutoff lands on the precision-recall curve",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "pr_curve_operating_points.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    y, proba, thresholds = load()
    print(f"Test rows: {len(y)}   surges: {int(y.sum())} ({y.mean():.3%})")
    for key, label, _ in POINTS:
        t = thresholds[key]
        pred = (proba >= t).astype(int)
        print(f"  {label:15} thr={t:.3f}  caught {int(((pred == 1) & (y == 1)).sum())}"
              f"/{int(y.sum())}  false alarms {int(((pred == 1) & (y == 0)).sum())}")

    outs = [plot_sweep(y, proba, thresholds),
            plot_confusions(y, proba, thresholds),
            plot_pr_points(y, proba, thresholds)]

    print("\nWrote:")
    for o in outs:
        print("  -", os.path.relpath(o, ROOT))


if __name__ == "__main__":
    main()
