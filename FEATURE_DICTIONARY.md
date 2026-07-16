# Feature Dictionary — Member 3: Feature Engineering (Anika)

Run `build_feature_matrix.py` after `join_and_build_features.py`. It reads
`model_ready_weekly.csv` / `model_ready_era2022.csv` and writes two files per
input: a **feature matrix** (safe model inputs) and a **targets** file (labels
only — never feed these into a model as X).

```
python build_feature_matrix.py
# -> feature_matrix_weekly.csv   + targets_weekly.csv
# -> feature_matrix_era2022.csv  + targets_era2022.csv   (use this one for modeling — see DATA_CLEANING.md)
```

## Features (`feature_matrix_*.csv`)

| Column | Source | Meaning |
|---|---|---|
| `state_territory` | Amy | 2-letter state code — identifier, not a numeric feature |
| `week_end` | Amy | Saturday ending the CDC epiweek — identifier |
| `log10_conc_mean` / `log10_conc_median` | Amy | Wastewater viral concentration that state-week, log10 |
| `conc_site_z_mean` | Amy | How unusual concentration is vs. each site's own baseline |
| `n_samples` / `n_sites` | Amy | Sample volume / site count backing the row (data-quality signal) |
| `pct_nondetect` | Amy | Fraction of samples below detection limit |
| `pop_served` | Amy | Population covered by contributing wastewater plants |
| `admits` | Amy | Hospital admissions **this** week (legitimate predictor — not future data) |
| `coverage` | Amy | Fraction of hospitals that reported that week |
| `admits_per100k` | Amy | Admissions normalized by population |
| `log10_conc_lag1/2/3` | Amy | Wastewater signal 1/2/3 weeks ago |
| `conc_delta_1w` | Amy | Week-over-week change in wastewater signal |
| `conc_roll3` | Amy | 3-week rolling average of wastewater signal |
| `month` | **Anika (new)** | Calendar month (1–12) from `week_end` |
| `epiweek_of_year` | **Anika (new)** | ISO week number (1–52) — finer-grained seasonality |
| `season_winter/spring/summer/fall` | **Anika (new)** | One-hot season, since respiratory illness is seasonal |
| `region_northeast/midwest/south/west/territory` | **Anika (new)** | One-hot US Census region grouped from state — coarser geography for states with thin data |
| `regime_mandatory_pre/mandatory_post` | Amy, encoded by Anika | Which hospital-reporting rule was active (one-hot) |
| `split_era` | Christal (era2022 file only) | `train`/`test` flag for the 2022+ split |

Dropped: `admits_this_week` (exact duplicate of `admits`).

## Targets (`targets_*.csv`)

Keyed by `state_territory` + `week_end`, joinable back to the feature matrix.
**Never use these as model inputs** — they're derived from next week's data.

| Column | Meaning |
|---|---|
| `admits_next_week` | Raw admissions count next week |
| `y_reg_next_admits` | Regression target (= `admits_next_week`) |
| `pct_change_next` | % change from this week to next week |
| `y_surge_next_week` | Classification target: 1 if next week rises >10% AND is above that state's median |
| `next_week_end` | Date of the predicted week (for reference/sanity checks) |

## Known gaps / left for the modeler

- `log10_conc_lag2` has 28 NaNs, `log10_conc_lag3` has 58 NaNs (era2022 matrix) — the first 1-2 weeks of each state's series have no prior lag data. Tree-based models (Anusha's likely first pass) handle NaN natively; linear models will need imputation or row-dropping.
- `state_territory` is left as a raw code, not one-hot encoded, since 50+ dummy columns is a modeling-stage decision (or use `region_*` instead for a lighter-weight geography signal).
