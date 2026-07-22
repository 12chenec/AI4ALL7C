# Final Model: COVID Surge Prediction

## Final model chosen: XGBoost

Selected based on `model_evaluation.md` and `evaluation/results/evaluation_metrics.csv`, which compared a majority-class baseline, Logistic Regression, Random Forest, and XGBoost on the 2022+ held-out test set (2,269 rows, 88 real surge weeks, 3.9% positive rate). XGBoost had the best overall ranking ability (highest ROC-AUC and PR-AUC) and the best F1 of the three real models, with a more balanced confusion matrix than Random Forest and far fewer false alarms than Logistic Regression.

## Files in this folder

| File | Contents |
|---|---|
| `surge_prediction_pipeline.pkl` | Final trained pipeline (joblib). Takes the raw merged feature/target dataframe, selects and orders the 17 model features, and runs the XGBoost classifier. |
| `pipeline_features.txt` | The 17 features the model uses, in order. |
| `pipeline_test_metrics.json` | Metrics from re-running the saved pipeline on the held-out test set. |
| `build_final_pipeline.py` | Script that built and saved the pipeline (loads data, fits, evaluates, saves). |

Preprocessing: no scaling or imputation is applied — XGBoost handles missing values natively, and the pipeline's only "preprocessing" step is selecting/ordering the 17 engineered features from a raw input row (via a `ColumnTransformer` passthrough). This mirrors how the model was trained and evaluated in `model_development/train_models.py` and `evaluation/evaluate_models.py`.

To load and use it:

```python
import joblib
pipe = joblib.load("final_model/surge_prediction_pipeline.pkl")
predictions = pipe.predict(new_data)          # 0/1 surge prediction
probabilities = pipe.predict_proba(new_data)[:, 1]   # surge probability
```

`new_data` just needs to contain the 17 columns in `pipeline_features.txt` (extra columns like `state_territory` or `week_end` are fine and get dropped automatically).

## Performance (2022+ held-out test set, default 0.5 threshold)

| Metric | Value |
|---|---|
| Accuracy | 0.957 |
| Balanced accuracy | 0.612 |
| Precision | 0.404 |
| Recall | 0.239 |
| Specificity | 0.986 |
| F1 | 0.300 |
| ROC-AUC | 0.869 |
| PR-AUC | 0.307 |
| Confusion matrix | TP 21, FP 31, FN 67, TN 2150 |

(Numbers reproduced directly by loading `surge_prediction_pipeline.pkl` and scoring the test split — match `results/model_results.csv` exactly. Minor run-to-run differences of a point or two on TP/FP vs. `evaluation/results/evaluation_metrics.csv` are normal XGBoost re-fit variance, not a preprocessing difference — both scripts use identical data, split, features, and hyperparameters.)

Accuracy is not the metric to trust here: the majority-class ("always no surge") baseline already scores 0.961 by construction, since only 3.9% of test weeks are true surges. ROC-AUC, PR-AUC, F1, and recall are the metrics that actually reflect surge-detection skill.

## Strengths

- Best overall ranking ability of the three models: highest ROC-AUC (0.869–0.871) and PR-AUC (0.30) on the test set, meaning its probability scores separate surge weeks from non-surge weeks better than Logistic Regression or Random Forest across nearly the whole threshold range.
- Best F1 score (0.30) among the real models, and the most balanced confusion matrix — it doesn't collapse to "always predict no surge" the way Random Forest effectively does (Random Forest recall is only 0.09).
- Outputs well-calibrated-enough probabilities to support threshold tuning: since the default 0.5 cutoff is conservative, moving the threshold down along the PR curve can trade some precision for meaningfully more recall.
- Handles the lag/rolling features' missing values natively, no imputation needed.

## Weaknesses

- Recall is still low at the default threshold (0.24–0.29), meaning it misses roughly 70% of real surge weeks out of the box. Not suitable as-is for an early-warning system without threshold tuning.
- Precision ceiling is modest (~0.40–0.53 depending on the specific re-fit): even when it flags a surge, it's wrong more often than not.
- Less interpretable than Logistic Regression; relies on the separate SHAP analysis for explainability.
- No model in this comparison is production-ready yet — even XGBoost's PR-AUC (~0.30) is far from what you'd want for a deployed alerting system, given the strong class imbalance (3.9% positive rate).

## Recommended next steps (carried over from `model_evaluation.md`)

1. Tune the decision threshold instead of using the default 0.5 — pick an operating point on the PR curve that matches the desired precision/recall trade-off.
2. Try resampling or `scale_pos_weight` to lift recall, the way `class_weight="balanced"` does for Logistic Regression.
3. Consider a regression framing on `y_reg_next_admits` as a second approach, since the binary surge label discards the magnitude of the jump.
