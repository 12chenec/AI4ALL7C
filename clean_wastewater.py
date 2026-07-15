"""
AI4ALL Group 7C - WASTEWATER CLEANING
Raw CDC wastewater CSV -> wastewater_weekly_clean.csv
Cleaning only. No joining, no features, no labels.

Every filter logs rows in/out to cleaning_log.csv so decisions are auditable.
"""
import pandas as pd, numpy as np, json, os

UP = '/mnt/user-data/uploads'
OUT = '/mnt/user-data/outputs'
os.makedirs(OUT, exist_ok=True)

LOG = []
def log(step, df, reason):
    LOG.append({'step': step, 'rows_remaining': len(df), 'reason': reason})
    print(f"[{step:38s}] rows={len(df):>7,}  | {reason}")

# ============================================================
# WASTEWATER CLEANING
# ============================================================
print("="*70); print("WASTEWATER CLEANING"); print("="*70)

WW_COLS = ['record_id','site','state_territory','county_fips','counties_served',
           'population_served','sample_collect_date','sample_type','sample_matrix',
           'sample_location','pcr_target','pcr_gene_target_agg','pcr_target_avg_conc',
           'pcr_target_units','lod_sewage','pcr_target_detect','pcr_type','flow_rate']

ww = pd.read_csv(f'{UP}/CDC_Wastewater_Data_for_SARS-CoV-2.csv', usecols=WW_COLS, low_memory=False)
log('00_raw_load', ww, 'as delivered by CDC')

# --- Step 1: parse dates ---
ww['sample_collect_date'] = pd.to_datetime(ww['sample_collect_date'], errors='coerce')
ww = ww.dropna(subset=['sample_collect_date'])
log('01_valid_dates', ww, 'drop unparseable sample_collect_date')

# --- Step 2: drop rows with no target concentration (the label-side feature) ---
ww = ww.dropna(subset=['pcr_target_avg_conc'])
log('02_conc_not_null', ww, 'pcr_target_avg_conc is our core feature; cannot impute a virus reading')

# --- Step 3: UNIT HARMONISATION (critical) ---
# Three unit systems exist. copies/g dry sludge is a DIFFERENT physical quantity
# (mass-normalised solids) and is NOT convertible to copies/L liquid.
ww['units_norm'] = ww['pcr_target_units'].str.strip().str.lower()

# 3a. log10 rows -> convert back to linear so all liquid rows share one scale
is_log = ww['units_norm'].str.startswith('log10', na=False)
print(f"    -> converting {is_log.sum():,} log10-reported rows to linear copies/L")
ww.loc[is_log, 'pcr_target_avg_conc'] = 10 ** ww.loc[is_log, 'pcr_target_avg_conc']
ww.loc[is_log, 'units_norm'] = 'copies/l wastewater'

# 3b. keep only liquid copies/L
ww = ww[ww['units_norm'] == 'copies/l wastewater'].copy()
log('03_units_copies_per_L', ww, 'drop copies/g dry sludge - incommensurable units, not convertible')

# --- Step 4: sample matrix comparability ---
# Keep the two dominant liquid matrices; drop sludge/effluent strays (n<300 combined)
KEEP_MATRIX = ['raw wastewater', 'post grit removal']
ww = ww[ww['sample_matrix'].isin(KEEP_MATRIX)]
log('04_matrix_filter', ww, f'keep {KEEP_MATRIX}; medians ~42k vs ~44k copies/L so comparable')

# --- Step 5: non-detects -> LOD/2 substitution ---
# 40,336 rows have conc==0 AND detect=='no'. Zero is not a true zero, it is
# "below limit of detection". log(0) is undefined, so substitute LOD/2 (standard
# environmental-chemistry convention) rather than dropping (would bias low weeks upward).
nd = (ww['pcr_target_avg_conc'] == 0)
has_lod = ww['lod_sewage'].notna()
ww['nondetect_flag'] = nd.astype(int)
ww.loc[nd & has_lod, 'pcr_target_avg_conc'] = ww.loc[nd & has_lod, 'lod_sewage'] / 2
# zeros with no LOD reported -> use global median LOD/2
med_lod = ww['lod_sewage'].median()
ww.loc[nd & ~has_lod, 'pcr_target_avg_conc'] = med_lod / 2
print(f"    -> {nd.sum():,} non-detects substituted with LOD/2 (median LOD={med_lod:,.0f})")
log('05_nondetect_LOD_half', ww, 'zeros replaced by LOD/2, flagged in nondetect_flag')

# --- Step 6: drop exact duplicate measurements ---
before = len(ww)
ww = ww.drop_duplicates(subset=['site','sample_collect_date','pcr_gene_target_agg','sample_matrix'],
                        keep='first')
log('06_dedupe', ww, f'removed {before-len(ww):,} dup site/date/gene/matrix rows')

# --- Step 7: log10 transform ---
# Concentrations span 0 -> 6.2e9 (9 orders of magnitude), heavily right-skewed.
# Log10 is standard for viral load and makes the feature usable by LogReg.
ww['log10_conc'] = np.log10(ww['pcr_target_avg_conc'].clip(lower=1))

# --- Step 8: outlier handling via per-site robust z-score ---
# Do NOT clip globally: a real outbreak IS a high value. Only flag values that are
# extreme *relative to that site's own history* (site-level assay differences are huge).
g = ww.groupby('site')['log10_conc']
med = g.transform('median')
mad = g.transform(lambda s: (s - s.median()).abs().median())
mad = mad.replace(0, np.nan)
ww['robust_z'] = 0.6745 * (ww['log10_conc'] - med) / mad
ww['outlier_flag'] = (ww['robust_z'].abs() > 5).astype(int)
print(f"    -> flagged {int(ww.outlier_flag.sum()):,} site-relative outliers (|robust z|>5), NOT dropped")
log('08_outlier_flagged', ww, 'flagged not dropped - spikes may be genuine outbreak signal')

# --- Step 9: site-level normalisation ---
# Different plants use different assays/methods -> raw levels not comparable across sites.
# Z-score within site so the model learns "high FOR THIS SITE" (matches proposal mitigation #1).
ww['conc_site_z'] = (ww['log10_conc'] - med) / g.transform('std').replace(0, np.nan)
ww['conc_site_z'] = ww['conc_site_z'].fillna(0)

# --- Step 10: weekly aggregation per state ---
# CDC epiweek convention: weeks run Sunday->Saturday, labelled by the SATURDAY end date.
# NHSN uses this. Wastewater must match or the join silently returns zero rows.
# weekday(): Mon=0..Sat=5, Sun=6. Days forward to next Saturday:
ww['week_end'] = ww['sample_collect_date'] + pd.to_timedelta(
    (5 - ww['sample_collect_date'].dt.weekday) % 7, unit='D')
ww['state_territory'] = ww['state_territory'].str.strip().str.lower()

weekly = (ww.groupby(['state_territory','week_end'])
            .agg(log10_conc_mean=('log10_conc','mean'),
                 log10_conc_median=('log10_conc','median'),
                 conc_site_z_mean=('conc_site_z','mean'),
                 n_samples=('record_id','count'),
                 n_sites=('site','nunique'),
                 pct_nondetect=('nondetect_flag','mean'),
                 pop_served=('population_served','sum'))
            .reset_index())
log('10_weekly_state_agg', weekly, 'aggregate site-days -> state-weeks')

# --- Step 11: coverage threshold ---
weekly = weekly[weekly['n_sites'] >= 2]
log('11_min_2_sites', weekly, 'drop state-weeks backed by a single site (unstable mean)')

weekly.to_csv(f'{OUT}/wastewater_weekly_clean.csv', index=False)

pd.DataFrame(LOG).to_csv(f'{OUT}/cleaning_log.csv', index=False)
print(f"\nWrote wastewater_weekly_clean.csv ({len(weekly):,} rows) + cleaning_log.csv")
