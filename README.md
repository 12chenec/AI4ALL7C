# AI4ALL Group 7C: Predicting COVID-19 Hospital Admission Surges from Wastewater

Can wastewater surveillance tell us a week in advance that COVID-19 hospital admissions are about to rise? This project builds an end-to-end machine learning pipeline to find out, from raw CDC data through to a deployed prediction app.

**Live app:** https://ai4all7c-svsehmuzxx4dv5jnwsjzbq.streamlit.app/

## The problem

For each US state and week, predict `y_surge_next_week`: will next week be a COVID admissions surge? A surge means admissions rise more than 10% over the current week **and** land above that state's median.

Only about 4% of weeks in the test period are surges, which shapes everything about how the models are evaluated. See "A note on accuracy" below.

## Data

Two CDC datasets, joined on `state_territory` and `week_end`:

| Source | What it gives us |
|---|---|
| CDC NWSS wastewater surveillance | Viral concentration in sewage, per treatment plant per sample |
| [CDC NHSN Weekly Hospital Respiratory Data](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data) (`ua7e-t2fy`) | Confirmed COVID admissions, per state per week |

Wastewater samples are aggregated to one row per state-week so the two sources line up. An earlier attempt used the CDC Case Surveillance dataset, but it is monthly, so a one-week prediction could not be checked against it: that left 220 usable labels versus 10,195 from the hospital data. It was dropped. Details in `DATA_CLEANING.md`.

Modeling uses the 2022+ subset (`feature_matrix_era2022.csv`), which excludes the volatile 2020 to 2021 period. The split is chronological: train on 2022-01 through 2025-07, test on 2025-07 through 2026-06.

## Repository layout

| Path | Contents |
|---|---|
| `data_cleaning/` | Cleaning scripts and cleaned weekly CSVs. See `DATA_CLEANING.md` |
| `eda/` | Exploratory analysis scripts |
| `feature_engineering/` | Feature construction. See `FEATURE_DICTIONARY.md` |
| `model_development/` | Model training and comparison (`train_models.py`) |
| `evaluation/` | Metrics, confusion matrices, ROC and PR curves. See `model_evaluation.md` |
| `final_model/` | The shipped pipeline, tuned thresholds. See `FINAL_MODEL.md` |
| `shap_analysis.py`, `shap/` | Explainability analysis and figures. See `SHAP_EXPLAINABILITY.md` |
| `app/app.py` | Streamlit prediction app |

## Running it

```bash
pip install -r requirements.txt

python final_model/build_final_pipeline.py   # trains the model, tunes thresholds
python evaluation/evaluate_models.py         # metrics and figures
python shap_analysis.py                      # explainability figures
streamlit run app/app.py                     # the app, at localhost:8501
```

Versions in `requirements.txt` are pinned to what the saved model was fitted with. If you upgrade scikit-learn or xgboost, rerun `build_final_pipeline.py` so the saved pipeline still loads.

## Results

XGBoost performed best of the models compared (Logistic Regression, Random Forest, XGBoost, against a majority-class baseline), reaching **ROC-AUC 0.873** and **PR-AUC 0.327** on the held-out test set, against a no-skill PR-AUC of 0.039.

The decision threshold turned out to matter more than the choice of model. At the default 0.5 cutoff the model catches only 21 of 88 surges; at a tuned threshold it catches 37, 46, or 60, depending on how many false alarms are acceptable. Thresholds are tuned on a validation window carved out of the training data, never on the test set. Full numbers are in `final_model/FINAL_MODEL.md`.

### A note on accuracy

Accuracy is misleading here and we do not lead with it. Because only 3.9% of test weeks are surges, a model that always predicts "no surge" scores 96% accuracy while catching zero surges. Recall, F1, balanced accuracy, ROC-AUC, and PR-AUC are the metrics that reflect real skill, and those are what the evaluation reports.

## Limitations

This is a course project and a research prototype, not a public-health tool. The main caveats: precision is low at useful recall levels, the surge base rate drifts sharply downward across the data (26% in 2022 to 1.7% in 2026), and the `admits_per100k` feature divides statewide admissions by the wastewater plants' population served, which produces impossible values. These are documented in `final_model/FINAL_MODEL.md`.

## Team

| Member | Area |
|---|---|
| Amy | Data cleaning |
| Christal | Data integration and EDA |
| Anika | Feature engineering |
| Anusha | Model development |
| Bhavya | Model evaluation |
| Janvi | Explainability and documentation |
