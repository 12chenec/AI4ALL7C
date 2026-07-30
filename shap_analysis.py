import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved from this file's location so the script runs from any directory.
# Note: this file must NOT be named shap.py, or `import shap` above would
# import it instead of the SHAP library.
ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(ROOT, "final_model", "surge_prediction_pipeline.pkl")
FEATURES_PATH = os.path.join(ROOT, "final_model", "pipeline_features.txt")
FEATURE_MATRIX_PATH = os.path.join(ROOT, "feature_matrix_era2022.csv")
TARGETS_PATH = os.path.join(ROOT, "targets_era2022.csv")
OUTPUT_DIR = os.path.join(ROOT, "shap")

TOP4_FOR_DEPENDENCE = ["admits_per100k", "conc_site_z_mean", "epiweek_of_year", "log10_conc_mean"]


def load_data_and_model():
    pipe = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH) as f:
        feat_cols = [line.strip() for line in f if line.strip()]

    fm = pd.read_csv(FEATURE_MATRIX_PATH)
    tg = pd.read_csv(TARGETS_PATH)
    df = fm.merge(tg, on=["state_territory", "week_end"], how="left")

    test = df[df.split_era == "test"].reset_index(drop=True)
    return pipe, feat_cols, test


def verify_metrics(pipe, test):
    """Re-score the pipeline on the real test set and print metrics for comparison
    against pipeline_test_metrics.json. Run this before trusting any SHAP output —
    if these don't match, the SHAP explanations below aren't for the shipped model."""
    y_test = test["y_surge_next_week"].astype(int)
    y_pred = pipe.predict(test)
    y_proba = pipe.predict_proba(test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("=== Reproduced test-set metrics (compare to pipeline_test_metrics.json) ===")
    print(f"accuracy:          {accuracy_score(y_test, y_pred):.4f}")
    print(f"balanced_accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"precision:         {precision_score(y_test, y_pred):.4f}")
    print(f"recall:            {recall_score(y_test, y_pred):.4f}")
    print(f"f1:                {f1_score(y_test, y_pred):.4f}")
    print(f"roc_auc:           {roc_auc_score(y_test, y_proba):.4f}")
    print(f"pr_auc:            {average_precision_score(y_test, y_proba):.4f}")
    print(f"tn, fp, fn, tp:    {tn}, {fp}, {fn}, {tp}")
    print()

    return y_pred, y_proba


def compute_shap(pipe, feat_cols, test):
    #Runs the trained XGBoost step through TreeExplainer on the preprocessed test rows.
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocessing"]

    X_test = pd.DataFrame(pre.transform(test), columns=feat_cols, index=test.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = float(np.array(explainer.expected_value).ravel()[0])

    return X_test, shap_values, expected_value


def make_global_figures(X_test, shap_values, feat_cols, outdir):
    # Beeswarm summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=len(feat_cols))
    plt.title("SHAP Summary — Feature Impact on Surge Prediction", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Mean |SHAP| bar chart
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=len(feat_cols))
    plt.title("Mean |SHAP value| — Global Feature Importance", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print the ranking table used in SHAP_EXPLAINABILITY.md
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
    print("=== Global feature importance (mean |SHAP|) ===")
    print(imp)
    print()


def make_dependence_figure(X_test, shap_values, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.flatten(), TOP4_FOR_DEPENDENCE):
        plt.sca(ax)
        shap.dependence_plot(feat, shap_values, X_test, ax=ax, show=False, interaction_index=None)
        ax.set_title(f"SHAP dependence: {feat}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_shap_dependence_top4.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_waterfall_figures(pipe, X_test, shap_values, expected_value, test, y_pred, y_proba, outdir):
    y_true = test["y_surge_next_week"].astype(int).values

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    if len(tp_idx) == 0 or len(fn_idx) == 0:
        print("Warning: could not find both a true positive and a false negative example.")
        return

    examples = [(tp_idx[0], "true_positive"), (fn_idx[0], "false_negative")]

    for idx, label in examples:
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_test.iloc[idx].values,
            feature_names=list(X_test.columns),
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=12)
        st = test.loc[idx, "state_territory"]
        wk = test.loc[idx, "week_end"]
        proba = y_proba[idx]
        plt.title(f"{label.replace('_', ' ').title()} — {st}, week {wk} (P={proba:.2f})", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{label}: state={st}, week={wk}, predicted probability={proba:.3f}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe, feat_cols, test = load_data_and_model()
    y_pred, y_proba = verify_metrics(pipe, test)

    X_test, shap_values, expected_value = compute_shap(pipe, feat_cols, test)

    make_global_figures(X_test, shap_values, feat_cols, OUTPUT_DIR)
    make_dependence_figure(X_test, shap_values, OUTPUT_DIR)
    make_waterfall_figures(pipe, X_test, shap_values, expected_value, test, y_pred, y_proba, OUTPUT_DIR)

    print(f"\nAll figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()