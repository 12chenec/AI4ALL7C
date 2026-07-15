"""
AI4ALL Group 7C - HOSPITAL ADMISSIONS CLEANING
Raw NHSN CSV -> hospital_admissions_weekly_clean.csv
Cleaning only. No joining, no features, no labels.

Source: CDC NHSN "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction"
Dataset ID: ua7e-t2fy   (do NOT use akn2-qxic / 82ci-krud / n3kj-exp9 - archived, stop May 2024)
Download:   https://data.cdc.gov/resource/ua7e-t2fy.csv?$limit=100000

Every filter logs rows in/out to cleaning_log_hospital.csv so decisions are auditable.
"""
import pandas as pd, numpy as np, os, sys, glob

UP, OUT = '/mnt/user-data/uploads', '/mnt/user-data/outputs'
os.makedirs(OUT, exist_ok=True)

LOG = []
def log(step, n, reason):
    LOG.append({'step': step, 'rows_remaining': n, 'reason': reason})
    print(f"[{step:34s}] rows={n:>7,} | {reason}")

# --- locate the NHSN file ---
cands = (glob.glob(f'{UP}/*nhsn*.csv') + glob.glob(f'{UP}/*hrd*.csv')
         + glob.glob(f'{UP}/*Weekly_Hospital*.csv') + glob.glob(f'{UP}/ua7e-t2fy*.csv'))
if not cands:
    sys.exit("NHSN file not found in uploads. See docstring for download URL.")
# multiple downloads may be present (e.g. a truncated 1k-row first attempt).
# Pick the largest file - the complete one.
NHSN = max(set(cands), key=os.path.getsize)
print("Using NHSN file:", NHSN)

print("="*70); print("HOSPITAL ADMISSIONS CLEANING"); print("="*70)

h = pd.read_csv(NHSN, low_memory=False)
h.columns = [c.strip().lower() for c in h.columns]
log('20_raw_load', len(h), 'as downloaded from data.cdc.gov')

# --- Step 1: parse dates, drop null admissions ---
# totalconfc19newadm is the actual measurement; it cannot be imputed.
h['week_end'] = pd.to_datetime(h['weekendingdate'], errors='coerce')
h = h.dropna(subset=['week_end'])
h['admits'] = pd.to_numeric(h['totalconfc19newadm'], errors='coerce')
h = h.dropna(subset=['admits'])
log('21_valid_date_and_admits', len(h), 'drop unparseable dates / null admissions (318 rows)')

# --- Step 2: drop aggregate rows that would double-count ---
# jurisdiction contains states AND rollups: "USA" and "Region 1".."Region 10".
# Rollups are sums of their member states. Left in, every state is counted 2-3x.
h['state_territory'] = h['jurisdiction'].str.strip().str.lower()
is_region = h['state_territory'].str.startswith('region', na=False)
h = h[~is_region]
h = h[~h['state_territory'].isin(['usa','us','national','all'])]
log('22_drop_rollups', len(h), 'drop USA + HHS Region aggregates (would double-count states)')

# --- Step 3: reporting-regime labelling ---
# Hospital reporting rules changed twice:
#   mandatory  : <= 2024-04-30
#   VOLUNTARY  : 2024-05-01 .. 2024-10-31  <- admissions dip because fewer hospitals
#                                             reported, NOT because fewer people got sick
#   mandatory  : >= 2024-11-01
VOL_START, VOL_END = pd.Timestamp('2024-05-01'), pd.Timestamp('2024-10-31')
h['regime'] = np.where(h.week_end < VOL_START, 'mandatory_pre',
               np.where(h.week_end <= VOL_END, 'voluntary', 'mandatory_post'))
print("\n  regime counts:\n", h.regime.value_counts().to_string())

# --- Step 4: coverage (data-quality weight, NOT a feature) ---
# % of hospitals that actually reported admissions that week.
# NHSN ships this pre-computed as a percentage (95.0 = 95% of hospitals reported).
if 'totalconfc19newadmperchosprep' in h.columns:
    h['coverage'] = pd.to_numeric(h['totalconfc19newadmperchosprep'], errors='coerce') / 100.0
elif 'numhosprep' in h.columns and 'numconfc19newadmhosprep' in h.columns:
    # fallback for the $select-trimmed download shape
    h['coverage'] = (pd.to_numeric(h['numconfc19newadmhosprep'], errors='coerce')
                     / pd.to_numeric(h['numhosprep'], errors='coerce').replace(0, np.nan))
else:
    raise KeyError("No coverage column found. Expected 'totalconfc19newadmperchosprep'. "
                   f"Got columns like: {[c for c in h.columns if 'hosprep' in c][:5]}")

# --- Step 5: drop the voluntary-reporting window ---
h = h[h['regime'] != 'voluntary'].copy()
log('23_drop_voluntary_window', len(h),
    'May-Oct 2024 voluntary reporting: admissions dip is an artifact, not epidemiology')

labels = h[['state_territory','week_end','admits','coverage','regime']].copy()
labels = labels.sort_values(['state_territory','week_end'])
labels.to_csv(f'{OUT}/hospital_admissions_weekly_clean.csv', index=False)

pd.DataFrame(LOG).to_csv(f'{OUT}/cleaning_log_hospital.csv', index=False)
print(f"\nWrote hospital_admissions_weekly_clean.csv ({len(labels):,} rows) + cleaning_log_hospital.csv")
print(f"  states/territories: {labels.state_territory.nunique()}")
print(f"  window: {labels.week_end.min().date()} -> {labels.week_end.max().date()}")
print(f"  low-coverage weeks (<80% hospitals reporting): {(labels.coverage<0.8).sum():,} "
      f"({(labels.coverage<0.8).mean()*100:.1f}%) - consider excluding, see checklist")
