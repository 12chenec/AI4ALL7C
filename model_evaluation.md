# Model Evaluation

My job was to evaluate the surge-prediction models, compare the baseline against the advanced ones with the right metrics (confusion matrices included), and write up where each one is strong and weak.

## What the models predict

It's binary classification. For each state and week we predict `y_surge_next_week`: is next week a COVID admissions surge? A surge is defined as next week rising more than 10% over this week AND landing above that state's median admissions.

I used the 2022+ dataset ([feature_matrix_era2022.csv](feature_matrix_era2022.csv) and [targets_era2022.csv](targets_era2022.csv)) with the `split_era` train/test flag. That's 6,860 training rows and 2,269 test rows across 17 features. Everything below is on the held-out test set.

To reproduce it, run `python evaluation/evaluate_models.py`. Outputs go to [evaluation/results/](evaluation/results/) and [evaluation/figures/](evaluation/figures/). The script re-fits the same models set up in [model_development/train_models.py](model_development/train_models.py) and saves the predicted probabilities, which you need for the ROC and PR curves.

## The big caveat: the classes are really imbalanced

Only 88 of the 2,269 test weeks (3.9%) are actual surges. This changes how you read every result. A model that just says "no surge" every week gets 96.1% accuracy and catches zero surges. Useless, but it looks great on accuracy alone. I put that exact model in the table below as the "Majority-class baseline" so nobody misses the point.

So accuracy is the wrong headline metric here. The ones that tell you whether we're catching surges are recall, precision, F1, balanced accuracy, ROC-AUC, and PR-AUC. Those are what to look at.

## Results

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Majority-class baseline (always "no surge") | 0.961 | 0.500 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Logistic Regression (baseline model) | 0.768 | 0.727 | 0.107 | 0.682 | 0.185 | 0.803 | 0.192 |
| Random Forest | 0.962 | 0.544 | 0.533 | 0.091 | 0.155 | 0.841 | 0.250 |
| XGBoost | 0.958 | 0.607 | 0.426 | 0.227 | 0.296 | 0.871 | 0.299 |

The PR-AUC no-skill baseline is the prevalence, 0.039, and the ROC-AUC no-skill baseline is 0.500. Full numbers with raw TP/FP/FN/TN counts are in [evaluation/results/evaluation_metrics.csv](evaluation/results/evaluation_metrics.csv).

### Confusion matrices

![Confusion matrices](evaluation/figures/confusion_matrices.png)

| Model | TN | FP | FN | TP | Read |
|---|---|---|---|---|---|
| Logistic Regression | 1,682 | 499 | 28 | 60 | Catches 60 of 88 surges but false-alarms 499 times |
| Random Forest | 2,174 | 7 | 80 | 8 | Almost never false-alarms but misses 80 of 88 surges |
| XGBoost | 2,154 | 27 | 68 | 20 | Middle ground, 20 caught against 27 false alarms |

## Baseline vs advanced

![Metric comparison](evaluation/figures/metric_comparison.png)

The accuracy column is where you get fooled. Random Forest's 0.962 is barely above the do-nothing baseline's 0.961, and it earns that almost entirely by predicting "no surge," which is right 96% of the time by default. Look one column over: its balanced accuracy is 0.544 (0.5 is a coin flip) and recall is 0.091, so it misses 91% of the real surges. High accuracy, near-useless model.

On the metrics that matter, the ranking depends on what you want the model to do. For overall ranking ability XGBoost wins, with the best ROC-AUC (0.871), PR-AUC (0.299), and F1 (0.296). In the ROC and PR plots its line sits above the others across almost the whole range, so if you tune its threshold you get the best precision/recall trade-off available. For actually catching surges Logistic Regression wins, with recall of 0.682 and balanced accuracy of 0.727, the highest of any model, flagging 60 of the 88 real surges. The `class_weight="balanced"` setting pushes it to favor catching positives.

![ROC curves](evaluation/figures/roc_curves.png)

![Precision-recall curves](evaluation/figures/pr_curves.png)

I put both ROC and PR curves in for a reason. Under heavy imbalance ROC-AUC can look flattering because the big pile of true negatives inflates it. The precision-recall curve is the more honest view: the dashed line at 0.039 is the no-skill floor, all three models clear it, but the precision ceiling of roughly 0.4 to 0.5 shows how hard this problem actually is.

## Strengths and weaknesses

### Logistic Regression (the baseline)

Highest recall (0.682) and balanced accuracy (0.727), so it actually catches most surges. Fully interpretable through its coefficients, fast, and a legitimate baseline the advanced models have to beat. The downside is precision of only 0.107, so about 89% of its surge alarms are false positives (499 of them). It also has the lowest ROC-AUC and PR-AUC, so the underlying ranking is the weakest. It buys recall by flagging almost anything borderline. Best when missing a surge costs way more than a false alarm, like a public-health early-warning system where you'd rather over-prepare.

### Random Forest

Highest precision (0.533), so when it does call a surge it's right about half the time, and it almost never false-alarms (7 total). Handles the NaN lag features and non-linear interactions without extra work. But it's the weakest model here despite the best accuracy. Recall of 0.091 and balanced accuracy of 0.544 mean it misses 80 of 88 surges, so as an early-warning tool it barely does anything. Its default 0.5 threshold is badly miscalibrated for this imbalance and it collapses toward "always no surge." Best when false alarms are expensive and you only want the most confident calls, or after threshold tuning and resampling.

### XGBoost (best overall)

Best F1 (0.296), ROC-AUC (0.871), and PR-AUC (0.299), so it's the strongest overall and has the best-balanced confusion matrix (20 TP, 27 FP, 68 FN). Its probability outputs give the most room to tune the threshold to whatever trade-off the team wants. Still only 0.227 recall at the default threshold though, so it misses 68 of 88 surges. Good ranking, conservative default cutoff. Less interpretable than logistic regression, but the SHAP analysis covers that, and it needs the OpenMP runtime (`libomp`) installed to run. Best when you want the single strongest model and can tune the threshold. This is the one I'd carry forward.

## Follow-up: tuning the threshold (the recommendation below, carried out)

The first recommendation in the original writeup was to stop using the default 0.5 cutoff. That has now been done in `final_model/build_final_pipeline.py`, so here is what it bought.

Nobody chose 0.5. It is the scikit-learn default that `.predict()` applies silently, and it assumes balanced classes and equal costs for a false positive and a false negative. Neither is true here. Thresholds were re-picked on a time-based validation window carved off the end of the training data (2024-12-14 onward, 1,502 rows). The test set was never used to choose one.

| Operating point | Threshold | Precision | Recall | F1 | Balanced acc. | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Default (unchosen) | 0.500 | 0.447 | 0.239 | 0.311 | 0.613 | 21 | 26 | 67 |
| High precision (F0.5) | 0.385 | 0.359 | 0.420 | 0.387 | 0.695 | 37 | 66 | 51 |
| Balanced (F1) | 0.241 | 0.199 | 0.523 | 0.288 | 0.719 | 46 | 185 | 42 |
| High recall (F2) | 0.145 | 0.124 | 0.682 | 0.209 | 0.743 | 60 | 425 | 28 |

Surges caught goes from 21 out of 88 up to 37, 46, or 60 depending on how many false alarms are tolerable. The 0.385 point is the interesting one: it beats the default on F1, recall, and balanced accuracy simultaneously, giving up only precision. ROC-AUC and PR-AUC are unchanged at 0.873 and 0.327, because moving the threshold does not change the model, only where the cutoff sits.

Two other things I tried that did **not** work, worth recording so nobody repeats them:

- **`scale_pos_weight`** shifts the operating point but does not improve ranking. ROC-AUC stayed at 0.872 versus 0.873 without it. It is redundant with threshold tuning.
- **Hyperparameter and training-window changes** did not help either. Depth 3 and 6, lower learning rate with more trees, subsampling, stronger regularization, training only on 2023+ or 2024+, and recency sample weighting were all tested. None improved PR-AUC over the current settings, and recency weighting made it worse (0.288 versus 0.327). The available gain here was in the threshold, not the model.

One finding that explains a lot of the difficulty: **the surge rate falls steadily over time**, from 26% of weeks in 2022 to 19% in 2023, 11% in 2024, 4.4% in 2025, and 1.7% in 2026. The model is trained on a period when surges were several times more common than in the test period, which is why a threshold tuned on validation transfers imperfectly to test.

## Bottom line

1. Report F1, ROC-AUC, PR-AUC, and recall as the headline numbers, not accuracy. Accuracy is dominated by the 96% "no surge" majority and makes a do-nothing model look strong.
2. XGBoost is the best overall, so it's the one carried forward into the SHAP work and the app.
3. Use a tuned threshold, not `.predict()`. This was the single biggest improvement available and it required no retraining.
4. Still not production-ready. PR-AUC is 0.327 against a no-skill floor of 0.039, which is real signal, but precision at useful recall is low. Remaining next steps:
   - Fix `admits_per100k` upstream. It divides statewide admissions by wastewater-plant population, mixing denominators, and produces impossible values such as Florida at 6,941 admissions per 100k in a single week. SHAP ranks it as an important feature, so this matters.
   - Address the base-rate drift, for example by calibrating probabilities on a recent window.
   - Consider a regression framing on `y_reg_next_admits`, since the binary label throws away the size of the jump.

## Files produced

| File | Contents |
|---|---|
| [evaluation/evaluate_models.py](evaluation/evaluate_models.py) | The evaluation script. Re-fits the models, captures probabilities, writes everything below. |
| [evaluation/results/evaluation_metrics.csv](evaluation/results/evaluation_metrics.csv) | Full metric table with TP/FP/FN/TN, balanced accuracy, specificity, PR-AUC. |
| [evaluation/results/predictions_with_proba.csv](evaluation/results/predictions_with_proba.csv) | Per-row test-set actual, prediction, and probability for each model. |
| [evaluation/results/confusion_matrices.txt](evaluation/results/confusion_matrices.txt) | Text confusion matrices with derived rates. |
| [evaluation/figures/confusion_matrices.png](evaluation/figures/confusion_matrices.png) | Confusion matrix per model. |
| [evaluation/figures/roc_curves.png](evaluation/figures/roc_curves.png) | ROC curves, all models on one plot. |
| [evaluation/figures/pr_curves.png](evaluation/figures/pr_curves.png) | Precision-recall curves, all models on one plot. |
| [evaluation/figures/metric_comparison.png](evaluation/figures/metric_comparison.png) | Grouped bar chart of the headline metrics. |
