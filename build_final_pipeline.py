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
    confusion_matrix,
)

ROOT = "/sessions/serene-pensive-fermat/mnt/AI4ALL7C-main"
FEATURE_FILE = os.path.join(ROOT, "feature_matrix_era2022.csv")
TARGET_FILE = os.path.join(ROOT, "targets_era2022.csv")
TARGET = "y_surge_next_week"

DROP_COLUMNS = [
    "state_territory", "week_end", "split_era",
    "admits_next_week", "y_reg_next_admits", "pct_change_next", "next_week_end",
    TARGET,
]

features = pd.read_csv(FEATURE_FILE)
targets = pd.read_csv(TARGET_FILE)
df = features.merge(targets, on=["state_territory", "week_end"], how="inner")

train_df = df[df["split_era"] == "train"].copy()
test_df = df[df["split_era"] == "test"].copy()

X_train = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
X_train = X_train.select_dtypes(include=np.number)
X_test = X_test[X_train.columns]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

print("Train rows:", len(X_train), "Test rows:", len(X_test), "Features:", X_train.shape[1])

FEATURE_LIST = list(X_train.columns)

# Final pipeline: a "select and order the 17 engineered feature columns"
# preprocessing step (passthrough, no scaling/imputation -- XGBoost handles
# missing values natively, same as the winning model in evaluation/evaluate_models.py)
# followed by the XGBoost classifier with the same hyperparameters used in
# model_development/train_models.py and evaluation/evaluate_models.py.
# This lets the pipeline accept a raw merged feature/target dataframe (with
# extra id/target columns) and predict directly.
preprocessing = ColumnTransformer(
    [("select_features", "passthrough", FEATURE_LIST)],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
    )),
])

# Fit on the full train_df (minus the target) so the ColumnTransformer selector
# is meaningful; it will pick out FEATURE_LIST regardless of extra columns present.
X_train_raw = train_df.drop(columns=[TARGET])
X_test_raw = test_df.drop(columns=[TARGET])

pipeline.fit(X_train_raw, y_train)
X_train, X_test = X_train_raw, X_test_raw

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

metrics = {
    "model": "XGBoost (final pipeline)",
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "specificity": specificity,
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "n_features": len(FEATURE_LIST),
}

print(json.dumps(metrics, indent=2))

OUT_DIR = "/sessions/serene-pensive-fermat/mnt/outputs/final_model"
os.makedirs(OUT_DIR, exist_ok=True)

joblib.dump(pipeline, os.path.join(OUT_DIR, "surge_prediction_pipeline.pkl"))

with open(os.path.join(OUT_DIR, "pipeline_features.txt"), "w") as f:
    for col in FEATURE_LIST:
        f.write(col + "\n")

with open(os.path.join(OUT_DIR, "pipeline_test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved pipeline + metrics to", OUT_DIR)
