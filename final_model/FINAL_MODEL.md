# Final Model: COVID Surge Prediction

## Final model chosen: XGBoost

Selected based on `model_evaluation.md` and `evaluation/results/evaluation_metrics.csv`, which compared a majority-class baseline, Logistic Regression, Random Forest, and XGBoost on the 2022+ held-out test set (2,269 rows, 88 real surge weeks, 3.9% positive rate). XGBoost had the best overall ranking ability (highest ROC-AUC and PR-AUC) and the best F1 of the three real models, with a more balanced confusion matrix than Random Forest and far fewer false alarms than Logistic Regression.

## Files in this folder

| File | Contents |
|---|---|
| `surge_prediction_pipeline.pkl` | Final trained pipeline (joblib). Takes the raw merged feature/target dataframe, selects and orders the 17 model features, and runs the XGBoost classifier. |
| `pipeline_features.txt` | The 17 features the model uses, in order. |
| `pipeline_thresholds.json` | The three tuned decision thresholds and how they were chosen. |
| `pipeline_test_metrics.json` | Test-set metrics at the default cutoff and at each tuned threshold. |
| `build_final_pipeline.py` | Script that builds everything above. Rerun it if the upstream data changes. |

Preprocessing: no scaling or imputation is applied. XGBoost handles missing values natively, and the pipeline's only preprocessing step is selecting and ordering the 17 engineered features from a raw input row (a `ColumnTransformer` passthrough). This mirrors how the model was trained and evaluated in `model_development/train_models.py` and `evaluation/evaluate_models.py`.

## The decision threshold matters more than the model here

`.predict()` applies a hardcoded 0.5 cutoff. Nobody chose that number, it is just the scikit-learn default, and it assumes balanced classes and equal costs for false positives and false negatives. Neither holds here: only 3.9% of test weeks are surges, and in an early-warning setting missing a surge is worse than a false alarm. At 0.5 the model flags almost nothing and misses 67 of 88 surges.

So `build_final_pipeline.py` now tunes the threshold. It carves a time-based validation window off the end of the training data (everything from 2024-12-14 onward, 1,502 rows), fits on the earlier data, and picks thresholds there. **The test set is never used to choose a threshold.** Three operating points are saved:

| Name | Threshold | Chosen by |
|---|---|---|
| `high_precision` | 0.385 | F0.5 on validation |
| `balanced` | 0.241 | F1 on validation |
| `high_recall` | 0.145 | F2 on validation |

## Performance (2022+ held-out test set)

Ranking quality is the same at every row below, since changing the threshold does not change the model: **ROC-AUC 0.873, PR-AUC 0.327** (no-skill PR-AUC would be 0.039).

| Operating point | Threshold | Accuracy | Balanced acc. | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Default (unchosen) | 0.500 | 0.959 | 0.613 | 0.447 | 0.239 | 0.311 | 21 | 26 | 67 |
| `high_precision` | 0.385 | 0.948 | 0.695 | 0.359 | 0.420 | **0.387** | 37 | 66 | 51 |
| `balanced` | 0.241 | 0.900 | 0.719 | 0.199 | 0.523 | 0.288 | 46 | 185 | 42 |
| `high_recall` | 0.145 | 0.800 | **0.743** | 0.124 | **0.682** | 0.209 | 60 | 425 | 28 |

The headline: moving off the default cutoff roughly doubles or triples the number of surges caught, from 21 out of 88 up to 37, 46, or 60 depending on how many false alarms you are willing to accept. The `high_precision` point is strictly better than the default on F1, recall, and balanced accuracy at once, and only gives up precision.

Accuracy falls as the threshold drops, and that is expected rather than a problem. The majority-class ("always no surge") baseline already scores 0.961 accuracy by construction, so accuracy mostly measures how often the model stays quiet. ROC-AUC, PR-AUC, recall, and balanced accuracy are the metrics that reflect actual surge-detection skill.

## Using it

```python
import joblib, json

pipe = joblib.load("final_model/surge_prediction_pipeline.pkl")
thresholds = json.load(open("final_model/pipeline_thresholds.json"))["thresholds"]

proba = pipe.predict_proba(new_data)[:, 1]
predictions = (proba >= thresholds["balanced"]).astype(int)
```

Use `predict_proba` and one of the saved thresholds rather than `pipe.predict()`, which silently reapplies the 0.5 default. `new_data` needs the 17 columns in `pipeline_features.txt`; extra columns like `state_territory` or `week_end` are fine and get dropped automatically.

## Strengths

- Best ranking ability of the models compared: ROC-AUC 0.873 and PR-AUC 0.327 on the test set, roughly 8x the no-skill PR-AUC of 0.039. Its probability scores separate surge weeks from quiet weeks well.
- With a tuned threshold it detects a useful share of surges. At the `high_recall` setting it catches 60 of 88 (68%).
- Handles the lag and rolling features' missing values natively, no imputation needed.
- Threshold is now an explicit, documented choice rather than an accident, and the Streamlit app exposes it so the trade-off is visible to whoever is using the tool.

## Weaknesses and honest limitations

- **Precision is low at useful recall levels.** Catching 68% of surges costs 425 false alarms across 2,269 weeks. This is a real constraint of the problem, not a tuning mistake: surges are rare, so most positive predictions are wrong.
- **Base-rate drift is severe and unaddressed.** The surge rate falls steadily across the data: 26% in 2022, 19% in 2023, 11% in 2024, 4.4% in 2025, 1.7% in 2026. The model is trained on a period when surges were roughly four times more common than in the test period, which is a large part of why the validation-tuned thresholds transfer imperfectly.
- **Hyperparameter tuning did not help.** Depth, learning rate, subsampling, regularization, recency-restricted training, and recency sample weighting were all tried; none improved PR-AUC over the current settings, and recency weighting made it worse. The gains available here came from the threshold, not the model.
- **`admits_per100k` is not trustworthy.** It divides statewide admissions by the wastewater plants' population served, which are different denominators. It produces impossible values (Florida at 6,941 admissions per 100k in one week, meaning 7% of the population). SHAP ranks it as an important feature, so this is worth fixing upstream in feature engineering before the numbers are relied on.
- Less interpretable than Logistic Regression, which is what the separate SHAP analysis is for.
- Not production-ready as a public-health alerting system. Treat it as a research prototype and one signal among several.

## Suggested next steps

1. Fix `admits_per100k` upstream so it uses actual state population, then retrain and see whether ranking improves.
2. Address the base-rate drift directly, for example by calibrating probabilities on a recent window or by predicting a state-relative surge definition that is stable over time.
3. Consider a regression framing on `y_reg_next_admits`, since the binary label discards the size of the jump.
