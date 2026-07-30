"""
Builds the final XGBoost pipeline and saves it into this folder.

Same model as train_models.py, except this also picks a decision threshold
instead of leaving it at the 0.5 default. Only about 4% of weeks are surges, so
0.5 is way too high and the model barely flags anything. We hold out the last
chunk of the training data as a validation set and pick the thresholds on that,
so the test set stays untouched.

Saves three options to pick from: high_precision, balanced, high_recall.

Writes surge_prediction_pipeline.pkl, pipeline_features.txt,
pipeline_thresholds.json and pipeline_test_metrics.json.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FEATURE_FILE = os.path.join(ROOT, "feature_matrix_era2022.csv")
TARGET_FILE = os.path.join(ROOT, "targets_era2022.csv")
OUT_DIR = HERE

TARGET = "y_surge_next_week"

DROP_COLUMNS = [
    "state_territory", "week_end", "split_era",
    "admits_next_week", "y_reg_next_admits", "pct_change_next", "next_week_end",
    TARGET,
]

# How much of the training weeks to fit on when tuning. The last 20% becomes
# the validation window.
VAL_SPLIT = 0.8


def load():
    features = pd.read_csv(FEATURE_FILE)
    targets = pd.read_csv(TARGET_FILE)
    df = features.merge(targets, on=["state_territory", "week_end"], how="inner")
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df


def make_pipeline(feature_list):
    # The "preprocessing" step just picks out the feature columns in the right
    # order, so we can hand the pipeline a full dataframe with id/target columns
    # still in it. No scaling or imputation since XGBoost handles NaN itself.
    preprocessing = ColumnTransformer(
        [("select_features", "passthrough", feature_list)],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )),
    ])


def best_threshold(y_true, proba, beta):
    """Best F-beta threshold. beta > 1 leans toward recall, beta < 1 toward precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # last point has no threshold attached, so drop it
    precision, recall = precision[:-1], recall[:-1]
    fbeta = (1 + beta ** 2) * precision * recall / (
        beta ** 2 * precision + recall + 1e-12
    )
    return float(thresholds[int(np.argmax(fbeta))])


def score(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    df = load()

    train_df = df[df["split_era"] == "train"].sort_values("week_end").copy()
    test_df = df[df["split_era"] == "test"].sort_values("week_end").copy()

    feature_list = list(
        train_df.drop(columns=DROP_COLUMNS, errors="ignore")
        .select_dtypes(include=np.number)
        .columns
    )

    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}  "
          f"Features: {len(feature_list)}")
    print(f"Surge rate  train {train_df[TARGET].mean():.3f}   "
          f"test {test_df[TARGET].mean():.3f}")

    # ---- pick thresholds on the validation window ----
    weeks = np.sort(train_df["week_end"].unique())
    cutoff = weeks[int(len(weeks) * VAL_SPLIT)]
    inner_df = train_df[train_df["week_end"] < cutoff]
    val_df = train_df[train_df["week_end"] >= cutoff]

    print(f"\nThreshold tuning: fit on {len(inner_df)} rows "
          f"(< {pd.Timestamp(cutoff).date()}), "
          f"tune on {len(val_df)} validation rows "
          f"(surge rate {val_df[TARGET].mean():.3f})")

    inner_pipe = make_pipeline(feature_list)
    inner_pipe.fit(inner_df.drop(columns=[TARGET]), inner_df[TARGET])
    val_proba = inner_pipe.predict_proba(val_df.drop(columns=[TARGET]))[:, 1]

    thresholds = {
        "high_precision": best_threshold(val_df[TARGET], val_proba, beta=0.5),
        "balanced": best_threshold(val_df[TARGET], val_proba, beta=1.0),
        "high_recall": best_threshold(val_df[TARGET], val_proba, beta=2.0),
    }
    for name, thr in thresholds.items():
        print(f"  {name:15} {thr:.4f}")

    # ---- refit on all the training data ----
    pipeline = make_pipeline(feature_list)
    pipeline.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])

    test_proba = pipeline.predict_proba(test_df.drop(columns=[TARGET]))[:, 1]
    y_test = test_df[TARGET]

    results = {
        "model": "XGBoost (final pipeline)",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_list),
        "test_surge_rate": float(y_test.mean()),
        "threshold_selection": {
            "method": "F-beta maximized on a time-based validation window "
                      "(last 20% of training weeks); the test set is never used "
                      "to choose a threshold",
            "validation_cutoff": str(pd.Timestamp(cutoff).date()),
            "n_validation": int(len(val_df)),
        },
        "by_threshold": {
            "default_0.5": score(y_test, test_proba, 0.5),
            **{name: score(y_test, test_proba, thr)
               for name, thr in thresholds.items()},
        },
    }

    print("\nTest-set performance:")
    header = f"  {'operating point':16} {'thr':>6} {'prec':>6} {'rec':>6} " \
             f"{'F1':>6} {'bal_acc':>8}   TP/FP/FN"
    print(header)
    for name, m in results["by_threshold"].items():
        print(f"  {name:16} {m['threshold']:6.3f} {m['precision']:6.3f} "
              f"{m['recall']:6.3f} {m['f1']:6.3f} {m['balanced_accuracy']:8.3f}   "
              f"{m['tp']}/{m['fp']}/{m['fn']}")

    # ---- save everything ----
    joblib.dump(pipeline, os.path.join(OUT_DIR, "surge_prediction_pipeline.pkl"))

    with open(os.path.join(OUT_DIR, "pipeline_features.txt"), "w") as fh:
        for col in feature_list:
            fh.write(col + "\n")

    with open(os.path.join(OUT_DIR, "pipeline_thresholds.json"), "w") as fh:
        json.dump({
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "default": "balanced",
            "selection": results["threshold_selection"],
        }, fh, indent=2)

    with open(os.path.join(OUT_DIR, "pipeline_test_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nSaved pipeline, features, thresholds and metrics to", OUT_DIR)


if __name__ == "__main__":
    main()
