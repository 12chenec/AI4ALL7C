# AI4ALL7C — Predicting COVID-19 Hospital Surges from Wastewater Data

AI4ALL Group 7C project. We predict next-week COVID-19 hospital admission surges by state, using SARS-CoV-2 concentrations measured in municipal wastewater as the leading signal.

## Why wastewater?

Wastewater surveillance catches infections before people show symptoms, get tested, or seek care — so it can lead hospital admissions by days to weeks. Unlike case counts, it doesn't depend on who bothered to get tested.

## Data sources

| Dataset | Grain | Source |
|---|---|---|
| CDC NWSS Wastewater Surveillance | 1 sample per site per day | [CDC Wastewater Data for SARS-CoV-2](https://data.cdc.gov/) |
| CDC NHSN Weekly Hospital Respiratory Data | 1 row per state per week | [Weekly Hospital Respiratory Data (HRD) Metrics](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data) |

**Note:** the original proposal used the [COVID-19 Case Surveillance dataset](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data), but its only time field is month-level, which produced just 220 usable weekly-aligned labels. We swapped to hospital admissions data (weekly, all 50 states) for 10,195 labels instead — see `DATA_CLEANING.md` for the full reasoning.

## Pipeline

```
clean_wastewater.py            # 584,287 rows -> 11,661 state-weeks
clean_hospital_admissions.py   # 20,703 rows  -> 15,632 state-weeks
        |
        v
join_and_build_features.py     # merge + lag/rolling features + train/test splits
        |
        v
build_feature_matrix.py        # temporal/geo features + leakage-safe feature/target split
        |
        v
feature_matrix_era2022.csv + targets_era2022.csv   <- modeling starts here
```

Run in order:
```bash
python data_cleaning/clean_wastewater.py
python data_cleaning/clean_hospital_admissions.py
python feature_engineering/join_and_build_features.py
python feature_engineering/build_feature_matrix.py
```

## Repo structure

```
AI4ALL7C/
├── README.md                          # you are here
├── DATA_CLEANING.md                   # cleaning decisions, null handling, the dataset swap
├── FEATURE_DICTIONARY.md              # what every column means (feature vs. target)
├── data_cleaning/
│   ├── clean_wastewater.py
│   ├── clean_hospital_admissions.py
│   ├── cleaning_log.csv               # audit trail: every filter, rows in/out, why
│   ├── cleaning_log_hospital.csv
│   ├── wastewater_weekly_clean.csv
│   └── hospital_admissions_weekly_clean.csv
├── feature_engineering/
│   ├── join_and_build_features.py
│   ├── build_feature_matrix.py
│   └── feature_build_log.csv
├── eda/
│   ├── merged_dataset_eda.py
│   └── wastewater_regional_analysis.py
├── model_ready_weekly.csv             # full timeline (2020+), features + targets mixed
├── model_ready_era2022.csv            # 2022+ only, more stable surge baseline
├── feature_matrix_era2022.csv         # <- final model inputs (no leakage)
├── targets_era2022.csv                # <- labels only, join on state_territory + week_end
├── feature_matrix_weekly.csv          # same split, full timeline version
├── targets_weekly.csv
└── cv_folds.csv                       # rolling-origin CV fold definitions
```

## Key modeling notes

- **Unit of analysis:** one state, one week. Both datasets are aggregated/reported at this grain so they can be joined on `state_territory` + `week_end` (CDC epiweek, Sunday→Saturday, labeled by the Saturday).
- **Targets:** `y_surge_next_week` (classification: does admissions rise >10% and land above that state's median next week?) and `y_reg_next_admits` (regression: raw admission count next week). Both live in `targets_*.csv`, never in the feature matrix.
- **Non-stationarity:** the surge base rate drops sharply over the pandemic (43% in 2020 → 2% in 2026), so `model_ready_era2022.csv` restricts to the more stable 2022+ era, and `cv_folds.csv` provides rolling-origin folds as the more honest evaluation method.
- **Known bias:** ~20% of US households are on septic systems and are invisible to wastewater monitoring. Hospital reporting was voluntary May–Oct 2024, artificially depressing admissions in that window (excluded from the clean data).

## Git Hub Page:
- https://12chenec.github.io/AI4ALL7C/
