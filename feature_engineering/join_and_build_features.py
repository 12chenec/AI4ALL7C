"""
AI4ALL Group 7C - JOIN + FEATURE ENGINEERING
wastewater_weekly_clean.csv + hospital_admissions_weekly_clean.csv -> model_ready_weekly.csv

NOT a cleaning script. Run the two cleaning scripts first:
    python clean_wastewater.py
    python clean_hospital_admissions.py
    python join_and_build_features.py   <- this file
"""
import pandas as pd, numpy as np, os

# OUT = '/mnt/user-data/outputs'
OUT = '/Users/christalchen/Documents/coding/AI4ALL7C'
LOG = []
def log(step, n, reason):
    LOG.append({'step': step, 'rows_remaining': n, 'reason': reason})
    print(f"[{step:34s}] rows={n:>7,} | {reason}")

print("="*70); print("JOIN + FEATURES"); print("="*70)


ww = pd.read_csv(f'{OUT}/wastewater_weekly_clean.csv', parse_dates=['week_end'])
log('30_wastewater_weekly', len(ww), 'from clean_wastewater.py')

labels = pd.read_csv(f'{OUT}/hospital_admissions_weekly_clean.csv', parse_dates=['week_end'])
log('30b_hospital_weekly', len(labels), 'from clean_hospital_admissions.py')



m = ww.merge(labels, on=['state_territory','week_end'], how='inner')
log('31_joined_weekly', len(m), 'inner join on state + WEEK (both weekly now)')

m = m.sort_values(['state_territory','week_end'])

# per-100k normalisation so states are comparable
m['admits_per100k'] = m['admits'] / (m['pop_served'] / 1e5)

# ---- TARGET: 1 week ahead (the proposal's actual horizon) ----
g = m.groupby('state_territory')
m['admits_next_week'] = g['admits'].shift(-1)
m['admits_this_week'] = m['admits']

# regression target
m['y_reg_next_admits'] = m['admits_next_week']

# classification target: "surge" = next week rises >10% AND is above that state's median.
# Threshold chosen empirically: >20% gave only ~8% positives (too rare to learn);
# >10% + above-median gives ~16%, a workable imbalance and still a defensible
# definition of a surge (rising meaningfully AND already above typical level).
SURGE_THRESHOLD = 0.10
state_med = g['admits'].transform('median')
m['pct_change_next'] = (m['admits_next_week'] - m['admits']) / m['admits'].replace(0, np.nan)
m['y_surge_next_week'] = ((m['pct_change_next'] > SURGE_THRESHOLD) &
                          (m['admits_next_week'] > state_med)).astype('Int64')
m.loc[m['admits_next_week'].isna(), 'y_surge_next_week'] = pd.NA

# guard: shift(-1) is only valid if the next row really is the next calendar week
m['next_week_end'] = g['week_end'].shift(-1)
gap_ok = (m['next_week_end'] - m['week_end']).dt.days == 7
m.loc[~gap_ok, ['y_reg_next_admits','y_surge_next_week']] = pd.NA
print(f"  voided {int((~gap_ok).sum()):,} rows where next row is not +7 days (gap in series)")

# lag features
for L in [1,2,3]:
    m[f'log10_conc_lag{L}'] = g['log10_conc_mean'].shift(L)
m['conc_delta_1w'] = m['log10_conc_mean'] - m['log10_conc_lag1']
m['conc_roll3'] = g['log10_conc_mean'].transform(lambda s: s.rolling(3, min_periods=2).mean())

model = m.dropna(subset=['y_surge_next_week','log10_conc_lag1']).copy()
log('32_model_ready', len(model), 'require label + >=1 lag')

print(f"\n  window: {model.week_end.min().date()} -> {model.week_end.max().date()}")
print(f"  states: {model.state_territory.nunique()}")
print(f"  INDEPENDENT weekly labels: {len(model):,}  (vs 220 monthly before)")
print(f"  surge base rate: {model.y_surge_next_week.mean():.1%}")

# ---- SPLIT ----
# NOTE: surge base rate declines sharply over time (2020: 43% -> 2026: 2%) because
# the pandemic itself wound down. A naive 80/20 time split puts ~18% positives in
# train and ~3% in test, so the model looks broken for reasons unrelated to wastewater.
# Two defensible options, both provided:
#
#  (a) era-restricted: drop the pre-vaccine 2020-2021 period, split within the
#      more stationary 2022+ era. Base rates are closer across the split.
#  (b) rolling-origin CV: several sequential train/test folds. Report mean +/- sd.
#      This is the honest way to evaluate a non-stationary time series.

era = model[model.week_end >= '2022-01-01'].copy()
cut = era.week_end.quantile(0.75)
tr, te = era[era.week_end <= cut], era[era.week_end > cut]
print("\n  --- (a) era-restricted split (2022+) ---")
print(f"  train: {len(tr):,} rows  base rate {tr.y_surge_next_week.mean():.1%}  (<= {pd.Timestamp(cut).date()})")
print(f"  test : {len(te):,} rows  base rate {te.y_surge_next_week.mean():.1%}  (>  {pd.Timestamp(cut).date()})")
era['split_era'] = np.where(era.week_end <= cut, 'train', 'test')

print("\n  --- (b) rolling-origin folds (use these for headline numbers) ---")
folds = []
yrs = [('2022-01-01','2023-01-01','2023-07-01'),
       ('2022-01-01','2023-07-01','2024-01-01'),
       ('2022-01-01','2024-01-01','2024-07-01'),
       ('2022-01-01','2025-01-01','2025-07-01')]
for i,(s,tr_end,te_end) in enumerate(yrs,1):
    a = model[(model.week_end>=s)&(model.week_end<tr_end)]
    b = model[(model.week_end>=tr_end)&(model.week_end<te_end)]
    if len(b)==0: continue
    folds.append({'fold':i,'train_n':len(a),'test_n':len(b),
                  'train_rate':round(a.y_surge_next_week.mean(),3),
                  'test_rate':round(b.y_surge_next_week.mean(),3),
                  'train_end':tr_end,'test_end':te_end})
    print(f"  fold {i}: train {len(a):>5,} ({a.y_surge_next_week.mean():.1%}) -> "
          f"test {len(b):>5,} ({b.y_surge_next_week.mean():.1%})")
pd.DataFrame(folds).to_csv(f'{OUT}/cv_folds.csv', index=False)

era.to_csv(f'{OUT}/model_ready_era2022.csv', index=False)
model.to_csv(f'{OUT}/model_ready_weekly.csv', index=False)
pd.DataFrame(LOG).to_csv(f'{OUT}/feature_build_log.csv', index=False)
print("\nWrote model_ready_weekly.csv")
