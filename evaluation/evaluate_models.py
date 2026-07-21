"""
Model evaluation for the surge-prediction models (Bhavya, Member 5).

This re-fits the same models Anusha set up in
model_development/train_models.py, on the same 2022+ data and the same
train/test split. The one thing it adds is that it saves each model's predicted
probabilities, which the committed results/predictions.csv doesn't have and
which we need in order to draw the ROC and precision-recall curves.

The target y_surge_next_week is very imbalanced (about 3.9% positives in the
test set), so accuracy on its own is misleading: always predicting "no surge"
already scores about 96%. To make that obvious the script also runs a
majority-class baseline and reports recall, F1, balanced accuracy, ROC-AUC and
PR-AUC alongside accuracy.

Outputs (under evaluation/):
  results/evaluation_metrics.csv      full metric table for every model
  results/predictions_with_proba.csv  per-row actual, prediction, probability
  results/confusion_matrices.txt      text confusion matrices with rates
  figures/confusion_matrices.png      one confusion matrix per model
  figures/roc_curves.png              ROC curves, all models on one plot
  figures/pr_curves.png               precision-recall curves, all models
  figures/metric_comparison.png       grouped bar chart of headline metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # write figures to file, no display needed
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as e:  # noqa: BLE001 - any import/runtime error means skip
    XGBOOST_AVAILABLE = False
    print("XGBoost unavailable, it will be skipped:", e)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FEATURE_FILE = os.path.join(ROOT, "feature_matrix_era2022.csv")
TARGET_FILE = os.path.join(ROOT, "targets_era2022.csv")

RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET = "y_surge_next_week"

# Columns that must never be model inputs (identifiers + future-derived targets).
# Mirrors Anusha's DROP_COLUMNS so the feature set is identical.
DROP_COLUMNS = [
    "state_territory",
    "week_end",
    "split_era",
    "admits_next_week",
    "y_reg_next_admits",
    "pct_change_next",
    "next_week_end",
    TARGET,
]

# Validated categorical palette (dataviz skill, first three slots pass all-pairs CVD).
COLORS = {
    "Logistic Regression": "#2a78d6",  # blue
    "Random Forest": "#eb6834",        # orange
    "XGBoost": "#1baf7a",              # aqua
    "Majority-class baseline": "#8a8a86",  # neutral gray (reference, not a real model)
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e2dd"


# ---------------------------------------------------------------------------
# Load, merge, split  (identical to train_models.py)
# ---------------------------------------------------------------------------
def load_data():
    features = pd.read_csv(FEATURE_FILE)
    targets = pd.read_csv(TARGET_FILE)
    df = features.merge(targets, on=["state_territory", "week_end"], how="inner")

    train_df = df[df["split_era"] == "train"].copy()
    test_df = df[df["split_era"] == "test"].copy()

    X_train = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")

    # keep only numeric columns, align test to train
    X_train = X_train.select_dtypes(include=np.number)
    X_test = X_test[X_train.columns]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    return train_df, test_df, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model zoo  (same configs as train_models.py, plus a majority-class baseline)
# ---------------------------------------------------------------------------
def build_models():
    models = {}

    # Majority-class reference: always predicts "no surge". Establishes the
    # accuracy floor that any useful model must beat on more than accuracy.
    models["Majority-class baseline"] = DummyClassifier(strategy="most_frequent")

    models["Logistic Regression"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    models["Random Forest"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )

    return models


# ---------------------------------------------------------------------------
# Fit + score
# ---------------------------------------------------------------------------
def evaluate(models, X_train, X_test, y_train, y_test):
    metrics = []
    proba = {}
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        preds[name] = y_pred
        proba[name] = y_proba

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        # ROC-AUC and PR-AUC are undefined for a constant predictor, so report NaN
        constant = len(np.unique(y_proba)) < 2
        roc = np.nan if constant else roc_auc_score(y_test, y_proba)
        pr = np.nan if constant else average_precision_score(y_test, y_proba)

        metrics.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "specificity": specificity,
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc,
                "pr_auc": pr,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    return pd.DataFrame(metrics), preds, proba


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_confusion_matrices(preds, y_test, model_order):
    real = [m for m in model_order if m != "Majority-class baseline"]
    n = len(real)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, real):
        cm = confusion_matrix(y_test, preds[name], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(name, fontsize=11, color=INK, pad=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No surge", "Surge"], fontsize=9)
        ax.set_yticklabels(["No surge", "Surge"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10, color=INK_SOFT)
        ax.set_ylabel("Actual", fontsize=10, color=INK_SOFT)

        thresh = cm.max() / 2.0
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if cm[i, j] > thresh else INK,
                    fontweight="bold",
                )
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(
        "Confusion matrices on the 2022+ test set",
        fontsize=12,
        color=INK,
        y=1.02,
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_roc(proba, y_test, model_order):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _style(ax)
    for name in model_order:
        if name == "Majority-class baseline":
            continue
        fpr, tpr, _ = roc_curve(y_test, proba[name])
        auc = roc_auc_score(y_test, proba[name])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
                label=f"{name}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.2, linestyle="--",
            label="Chance (AUC 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=11, color=INK)
    ax.set_ylabel("True positive rate (recall)", fontsize=11, color=INK)
    ax.set_title("ROC curves", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_pr(proba, y_test, model_order):
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _style(ax)
    for name in model_order:
        if name == "Majority-class baseline":
            continue
        prec, rec, _ = precision_recall_curve(y_test, proba[name])
        ap = average_precision_score(y_test, proba[name])
        ax.plot(rec, prec, color=COLORS[name], linewidth=2,
                label=f"{name}  (PR-AUC {ap:.3f})")
    ax.axhline(prevalence, color=INK_SOFT, linewidth=1.2, linestyle="--",
               label=f"Baseline = prevalence ({prevalence:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=11, color=INK)
    ax.set_ylabel("Precision", fontsize=11, color=INK)
    ax.set_title("Precision-recall curves", fontsize=12, color=INK,
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "pr_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_metric_comparison(metrics_df, model_order):
    shown = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    nice = ["Accuracy", "Balanced\naccuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    real = [m for m in model_order if m != "Majority-class baseline"]

    x = np.arange(len(shown))
    width = 0.8 / len(real)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    for i, name in enumerate(real):
        row = metrics_df[metrics_df["model"] == name].iloc[0]
        vals = [row[m] for m in shown]
        offset = (i - (len(real) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=name,
                      color=COLORS[name], edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11, color=INK)
    ax.set_title("Model performance by metric", fontsize=12, color=INK,
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5, ncol=len(real), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "metric_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def write_confusion_text(metrics_df, y_test):
    lines = []
    lines.append("CONFUSION MATRICES & DERIVED RATES - 2022+ test set")
    lines.append("=" * 60)
    lines.append(f"Test rows: {len(y_test):,}   "
                 f"Actual surges (positives): {int(y_test.sum()):,} "
                 f"({y_test.mean() * 100:.2f}%)")
    lines.append("")
    lines.append("Layout:            Predicted")
    lines.append("                 No surge   Surge")
    lines.append("")
    for _, r in metrics_df.iterrows():
        lines.append(f"### {r['model']}")
        lines.append(f"Actual No surge   {r['tn']:>7,}  {r['fp']:>6,}")
        lines.append(f"Actual Surge      {r['fn']:>7,}  {r['tp']:>6,}")
        lines.append(
            f"  recall(sens)={r['recall']:.3f}  specificity={r['specificity']:.3f}  "
            f"precision={r['precision']:.3f}  F1={r['f1']:.3f}"
        )
        roc = "n/a" if pd.isna(r["roc_auc"]) else f"{r['roc_auc']:.3f}"
        pr = "n/a" if pd.isna(r["pr_auc"]) else f"{r['pr_auc']:.3f}"
        lines.append(f"  ROC-AUC={roc}  PR-AUC={pr}  "
                     f"accuracy={r['accuracy']:.3f}  balanced_acc={r['balanced_accuracy']:.3f}")
        lines.append("")
    out = os.path.join(RESULTS_DIR, "confusion_matrices.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    train_df, test_df, X_train, X_test, y_train, y_test = load_data()
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}   "
          f"features: {X_train.shape[1]}")
    print(f"  test surge rate: {y_test.mean() * 100:.2f}% "
          f"({int(y_test.sum())} of {len(y_test)})")

    models = build_models()
    print("Models:", ", ".join(models.keys()))

    metrics_df, preds, proba = evaluate(models, X_train, X_test, y_train, y_test)

    # order: baseline first, then models by F1 desc for readability
    order = ["Majority-class baseline"] + (
        metrics_df[metrics_df["model"] != "Majority-class baseline"]
        .sort_values("f1", ascending=False)["model"]
        .tolist()
    )
    metrics_df["model"] = pd.Categorical(metrics_df["model"], categories=order, ordered=True)
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)

    # ---- save metric table
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMetrics:\n", metrics_df.to_string(index=False))

    # ---- save per-row predictions + probabilities
    pred_out = pd.DataFrame(
        {
            "state_territory": test_df["state_territory"].values,
            "week_end": test_df["week_end"].values,
            "actual": y_test.values,
        }
    )
    for name in order:
        if name == "Majority-class baseline":
            continue
        pred_out[f"{name}_pred"] = preds[name]
        pred_out[f"{name}_proba"] = np.round(proba[name], 6)
    pred_path = os.path.join(RESULTS_DIR, "predictions_with_proba.csv")
    pred_out.to_csv(pred_path, index=False)

    # ---- text confusion report
    cm_txt = write_confusion_text(metrics_df, y_test)

    # ---- figures
    f1 = plot_confusion_matrices(preds, y_test, order)
    f2 = plot_roc(proba, y_test, order)
    f3 = plot_pr(proba, y_test, order)
    f4 = plot_metric_comparison(metrics_df, order)

    print("\nWrote:")
    for p in [metrics_path, pred_path, cm_txt, f1, f2, f3, f4]:
        print("  -", os.path.relpath(p, ROOT))
    print("\nDone.")


if __name__ == "__main__":
    main()
