import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# -----------------------------
# File paths
# -----------------------------

FEATURE_FILE = "feature_matrix_era2022.csv"
TARGET_FILE = "targets_era2022.csv"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

print("Loading data...")

features = pd.read_csv(FEATURE_FILE)
targets = pd.read_csv(TARGET_FILE)

print("Features:", features.shape)
print("Targets:", targets.shape)


# -----------------------------
# Merge features + targets
# -----------------------------

df = features.merge(
    targets,
    on=["state_territory", "week_end"],
    how="inner"
)

print("Merged dataset:", df.shape)


# -----------------------------
# Split using Anika's split_era
# -----------------------------

train_df = df[df["split_era"] == "train"]
test_df = df[df["split_era"] == "test"]

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# -----------------------------
# Prepare X and y
# -----------------------------

TARGET = "y_surge_next_week"


DROP_COLUMNS = [
    "state_territory",
    "week_end",
    "split_era",

    # future-derived targets
    "admits_next_week",
    "y_reg_next_admits",
    "pct_change_next",
    "next_week_end",

    # classification target
    TARGET
]


X_train = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
y_test = test_df[TARGET]


# Convert any remaining non-numeric columns
X_train = X_train.select_dtypes(include=np.number)
X_test = X_test[X_train.columns]


print("Model input features:", X_train.shape[1])


# -----------------------------
# Evaluation helper
# -----------------------------

results = []
all_predictions = pd.DataFrame({
    "state_territory": test_df["state_territory"],
    "week_end": test_df["week_end"],
    "actual": y_test
})


def evaluate_model(name, model):

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions


    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)


    cm = confusion_matrix(y_test, predictions)


    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("AUC:", auc)
    print("Confusion Matrix:")
    print(cm)


    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })


    all_predictions[name + "_prediction"] = predictions


# -----------------------------
# Baseline Model
# -----------------------------

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


evaluate_model(
    "Logistic Regression",
    baseline
)


# -----------------------------
# Random Forest
# -----------------------------

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])


evaluate_model(
    "Random Forest",
    rf
)


# -----------------------------
# XGBoost
# -----------------------------

if XGBOOST_AVAILABLE:

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )

    evaluate_model(
        "XGBoost",
        xgb
    )

    # Save trained XGBoost model for SHAP analysis
    joblib.dump(
        xgb,
        "results/xgboost_model.pkl"
    )

    print("Saved XGBoost model for explainability.")

else:
    print("\nXGBoost not installed. Skipping.")


# -----------------------------
# Save results
# -----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_results.csv"),
    index=False
)

all_predictions.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)


# Save feature names for SHAP
with open("results/model_features.txt", "w") as f:
    for col in X_train.columns:
        f.write(col + "\n")


print("\nFinished!")
print("Saved:")
print(" - results/model_results.csv")
print(" - results/predictions.csv")
print(" - results/xgboost_model.pkl")
print(" - results/model_features.txt")