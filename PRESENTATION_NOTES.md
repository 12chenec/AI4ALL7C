# Presentation notes: Model Evaluation and Explainability (Bhavya)

Talking points and figures for my four slides. Numbers all come from the 2022+
held-out test set (2,269 state-weeks, 88 real surges). Everything here is
reproducible with `python evaluation/evaluate_models.py`,
`python evaluation/threshold_analysis.py` and `python shap_analysis.py`.

---

## Slide 1: Confusion matrix and the metrics that matter

**Show:** `evaluation/figures/confusion_matrices.png`, then
`evaluation/figures/metric_comparison.png`

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Always "no surge" | 0.961 | 0.500 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Logistic Regression | 0.768 | 0.727 | 0.107 | 0.682 | 0.185 | 0.803 | 0.192 |
| Random Forest | 0.962 | 0.544 | 0.533 | 0.091 | 0.155 | 0.841 | 0.250 |
| XGBoost | 0.958 | 0.607 | 0.426 | 0.227 | 0.296 | 0.871 | 0.299 |

Confusion matrices at the default cutoff:

| Model | TN | FP | FN | TP | One-line read |
|---|---|---|---|---|---|
| Logistic Regression | 1,682 | 499 | 28 | 60 | Catches 60 of 88, but cries wolf 499 times |
| Random Forest | 2,174 | 7 | 80 | 8 | Almost never false-alarms, misses 80 of 88 |
| XGBoost | 2,154 | 27 | 68 | 20 | The middle ground |

**Say:** XGBoost is the model we shipped. It has the best ranking ability of the
three, ROC-AUC 0.87 and PR-AUC 0.33 against a no-skill floor of 0.039, so it is
roughly 8 times better than guessing at ordering surge weeks above quiet ones.
Logistic Regression catches the most surges but at 89% false alarms. Random
Forest looks best on accuracy and is actually the weakest, which leads into the
next slide.

---

## Slide 2: Why accuracy alone is misleading

**Show:** the pie chart already in the deck (slide 14), then
`evaluation/figures/metric_comparison.png`

**The number to lead with:** only 88 of 2,269 test weeks are surges, 3.9%.

**Say:** A model that predicts "no surge" every single week scores **96.1%
accuracy** and catches **zero** surges. That is the first row of the table, and
it is not a real model, it is a constant. So 96% accuracy is the floor, not an
achievement.

Random Forest scores 96.2% accuracy. That is 0.1 points above the do-nothing
baseline. Look at balanced accuracy instead and it is 0.544, where 0.5 is a coin
flip, and its recall is 0.091, so it misses 91% of real surges. High accuracy,
near-useless model.

**The fix:** report recall, precision, F1, balanced accuracy, ROC-AUC and PR-AUC.
PR-AUC especially, because under heavy imbalance ROC-AUC is flattered by the
2,181 easy true negatives, while PR-AUC compares against the 3.9% floor.

**Bonus if there is time:** `evaluation/figures/confusion_by_threshold.png`.
Nobody picked the 0.5 cutoff, it is just the scikit-learn default and it assumes
balanced classes. Retuning it on a validation window (never on the test set)
takes surges caught from 21 of 88 up to 37, 46 or 60 depending on how many false
alarms you accept. Same model, same probabilities, just a different cutoff.

---

## Slide 3: SHAP and feature importance

**Show:** `shap/fig_shap_importance_bar.png`

| Rank | Feature | Mean abs SHAP | What it is |
|---|---|---|---|
| 1 | `admits_per100k` | 0.817 | Current admissions, population-normalized |
| 2 | `conc_site_z_mean` | 0.800 | Wastewater level vs that site's own baseline |
| 3 | `epiweek_of_year` | 0.529 | Week of the year, so seasonality |
| 4 | `log10_conc_mean` | 0.453 | Raw wastewater concentration |
| 5 | `log10_conc_lag3` | 0.236 | Wastewater 3 weeks ago |

**Say:** Current hospital load and the wastewater z-score dominate, together
about a third of total feature impact. The most interesting result is that the
**normalized** wastewater signal beats the **raw** one. `conc_site_z_mean` ranks
2nd while `log10_conc_mean` is 4th and `conc_roll3` is dead last of 17. The model
cares whether a reading is unusual *for that specific plant*, not whether the raw
number is big, which is exactly what the cleaning step was designed for since raw
concentrations are not comparable across labs and plants.

Seasonality also outranks every individual wastewater lag.

---

## Slide 4: Which variables raise or lower predicted surge risk

**Show:** `shap/fig_shap_summary.png` (the beeswarm). Red is a high feature
value, blue is low. Right of centre pushes toward surge.

Backed by `shap/feature_direction.csv`, which correlates each feature's value
against its SHAP value, so the direction is a number rather than an eyeball call.

**Pushes risk UP when high:**

| Feature | Correlation | Reading |
|---|---|---|
| `epiweek_of_year` | +0.94 | Later in the year raises risk, winter respiratory season |
| `conc_site_z_mean` | +0.92 | The more unusual the wastewater is for that site, the higher the risk |
| `log10_conc_mean` | +0.74 | Higher raw concentration raises risk |
| `conc_delta_1w` | +0.65 | Week-over-week rise raises risk |
| `admits_per100k` | +0.53 | Hospitals already loaded raises risk |

**Pushes risk DOWN when high:**

| Feature | Correlation | Reading |
|---|---|---|
| `log10_conc_lag3` | -0.77 | High levels 3 weeks ago *lower* today's risk |
| `n_sites` | -0.78 | More reporting plants lowers predicted risk |
| `coverage` | -0.51 | Better hospital reporting lowers predicted risk |

**The best point to make here:** look at the lag features together.
`log10_conc_lag1` is **+0.66** but `log10_conc_lag3` is **-0.77**. The model is
not reacting to the level of the virus, it is reacting to the *slope*. High now,
low three weeks ago means the signal is climbing, and that is what it flags as a
surge. That matches the EDA finding that admissions track the wastewater curve
about a week behind.

**The honest caveat to include:** `coverage`, `n_sites` and `n_samples` are all
data-quality columns, not biology, and all three say the same thing, that thinner
monitoring means higher predicted risk. The model has partly learned "when we
know less, predict surge." That is confounding rather than epidemiology, and it
lines up with the regional bias point on slide 21.

Also worth stating plainly if asked: **`admits_per100k`, the number one feature,
is computed wrong.** It divides statewide admissions by the population served by
the wastewater plants, which are two different denominators. It produces
impossible values, for example Florida at 6,941 admissions per 100k in one week,
which would be 7% of the state hospitalized. Fixing it is the top item for future
work.

---

## Things in the deck that should be corrected before presenting

Found these while pulling the numbers together.

1. **Slide 9 lists only Logistic Regression and Random Forest.** The model we
   actually shipped is XGBoost, and it is what the evaluation, SHAP analysis and
   app all use. XGBoost needs to be on that slide.
2. **Slide 15 says the wastewater data was "joined with CDC Case Surveillance
   data for outbreak labels."** That contradicts slide 7, which correctly says
   Case Surveillance was dropped. The labels come from the hospital admissions
   dataset.
3. **Slides 16 and 18 say the dataset stopped updating in July 2024 and that we
   limit training and testing to before that date.** That is left over from the
   proposal. We actually train on 2022-01 to 2025-07 and test on 2025-07 to
   2026-06.
4. **Slide 15 says "541K+ wastewater samples."** `DATA_CLEANING.md` flags this as
   needing to be 584,287.
5. **Slide 13 shows a different app** than the one in the repo. Worth checking
   which version is actually deployed before demoing it.

---

## Figure index

| File | Use |
|---|---|
| `evaluation/figures/confusion_matrices.png` | Slide 1, per-model confusion matrices |
| `evaluation/figures/metric_comparison.png` | Slides 1 and 2, metric bar chart |
| `evaluation/figures/roc_curves.png` | Backup, ranking quality |
| `evaluation/figures/pr_curves.png` | Backup, the honest view under imbalance |
| `evaluation/figures/confusion_by_threshold.png` | Slide 2 bonus, 21 to 60 surges caught |
| `evaluation/figures/threshold_sweep.png` | Backup, precision/recall trade-off |
| `evaluation/figures/pr_curve_operating_points.png` | Backup, where each cutoff sits |
| `shap/fig_shap_importance_bar.png` | Slide 3, global importance |
| `shap/fig_shap_summary.png` | Slide 4, direction of effect |
| `shap/fig_shap_dependence_top4.png` | Backup, shape of the top 4 effects |
| `shap/fig_waterfall_true_positive.png` | Backup, a caught surge |
| `shap/fig_waterfall_false_negative.png` | Backup, a missed surge |
| `shap/feature_direction.csv` | Slide 4, the direction numbers |
