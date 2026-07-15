# Member 1: Data Cleaning — Amy

## ⚠️ Headline: we had to swap a dataset

**The COVID Case Surveillance dataset didn't work.** Its only time column is `case_month` — it tells you a case happened in "January 2024," not which week. And it only covers Jan–Jun 2024 (6 months).

You can't check a 1-week prediction against a monthly number. When I joined it to the wastewater data, the case count was *identical across all 4 weeks of each month* — the label physically doesn't move week to week. That left **220 usable labels**.

**Replaced with:** CDC NHSN Weekly Hospital Respiratory Data (`ua7e-t2fy`).
https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
Weekly, all 50 states, Aug 2020 – Jul 2026 → **10,195 labels instead of 220.**

**This changes our research question:** we now predict a surge in *hospital admissions*, not *cases*. Arguably better (admissions don't depend on who bothered to get tested).

**TODO for the team:** change the Datasets slide so it doesn't list Case Surveillance. Proposal also says "541K samples" should be 584,287.

---

## Cleaning checklist

| # | Check | Result |
|---|---|---|
| 1 | Profile both files first | WW 584,287 × 38 · NHSN 20,703 × 322 |
| 2 | Null values | see below |
| 3 | Data types | see below |
| 4 | Duplicates | 3,468 dropped from WW · 0 in NHSN |
| 5 | Units / measurement consistency | ~100K rows dropped (wrong units) |
| 6 | Aggregate rows that double-count | 3,717 dropped from NHSN |
| 7 | Range/validity (negatives, future dates) | all clean ✅ |
| 8 | Log every step | `cleaning_log.csv`, `cleaning_log_hospital.csv` |

---

## Null values

**Wastewater:** dropped only the 201 rows missing `pcr_target_avg_conc`. That's the actual virus measurement — you can't invent it, and 201 rows is 0.03%.

Other columns are 19–46% null (`flow_rate`, `inhibition_adjust`, etc). **Left them alone — we don't use them.** Important: never run `dropna()` across the whole dataframe. It would take us from 584K rows to nearly zero because of columns we don't even touch.

**Hospital:** dropped 318 rows with no admission count.

**Case surveillance (worth knowing):** its nulls are *hidden* behind the strings `"Missing"` and `"Unknown"`. `.isna()` reports 0% null for `icu_yn` — but it's actually **98% unusable**. Always check for sentinel values, not just nulls.

---

## Data types

| Issue | Fix |
|---|---|
| `sample_collect_date` loads as a string | `pd.to_datetime()` — 0 failures |
| `weekendingdate` loads as a string | `pd.to_datetime()` — 0 failures |
| **`county_fips` must stay a string** | `"02020"` is Alaska. Convert to int → `2020` → silently corrupted |
| `pcr_target_avg_conc`, `population_served` | already numeric ✅ |

---

## Duplicates

`record_id` is unique, so a naive duplicate check finds **nothing**. The real duplicates are **repeat measurements of the same site + day + gene + matrix** — 3,468 dropped, keeping the first.

NHSN had zero duplicates.

---

## Wastewater cleaning — `clean_wastewater.py`

**584,287 rows → 11,661 state-weeks**

Main column: `pcr_target_avg_conc` (high = lots of infected people in that sewer system).

Key decisions:

1. **Dropped ~100K rows measured in `copies/g dry sludge`.** The units column mixes three systems. Dry sludge is a *different physical quantity* than `copies/L wastewater` — median ~1.3M vs ~42K, a 30× gap. Not convertible. Mixing them would create fake outbreak spikes wherever a sludge site reports.
2. **40,336 zeros weren't really zeros.** Every one has `pcr_target_detect = 'no'` — they mean "below the detection limit," not "no virus." Dropping them would bias quiet weeks upward; keeping 0 breaks `log10(0)`. Replaced with LOD/2 (standard environmental-chemistry practice) and flagged.
3. **Log10-transformed** — raw values span 0 to 6.2 billion. Way too skewed for logistic regression.
4. **Flagged outliers but did NOT delete them.** A real outbreak *is* a high value. Deleting extremes would delete exactly what we're trying to detect. 930 flagged, all kept.
5. **Aggregated to state-weeks.** Raw data is one row per sample per day; hospital data is one row per state per week. They have to match to join. All of Ohio's samples in week X → one Ohio row for week X.
6. **Required ≥2 sites per state-week** — a state-week backed by one plant is too unstable.

### Output: `wastewater_weekly_clean.csv` (11,661 rows, 0 nulls)

| column | meaning |
|---|---|
| `state_territory` | 2-letter code, lowercased to match NHSN |
| `week_end` | **Saturday** ending the CDC epiweek |
| `log10_conc_mean` | avg concentration that state-week, log10 (6.588 ≈ 3,870,000 copies/L) |
| `log10_conc_median` | median instead of mean — robust to one weird plant |
| `conc_site_z_mean` | **is this high *for these specific sites*?** 0 = typical, +1.85 = well above their own normal, −1 = below |
| `n_samples` | how many lab samples fed this row |
| `n_sites` | how many distinct treatment plants (more = more trustworthy) |
| `pct_nondetect` | fraction of samples below detection (0 = all found virus, 1 = none did) |
| `pop_served` | people covered, summed across plants |

---

## Hospital cleaning — `clean_hospital_admissions.py`

**20,703 rows → 15,632 state-weeks**

1. **Dropped `USA` and `Region 1`–`Region 10` rows** — those are totals *of* the states. Left in, every state gets counted 2–3×.
2. **Dropped May–Oct 2024.** Hospital reporting was **voluntary** then (mandatory before and after). Admissions dip because fewer hospitals filed paperwork, not because fewer people got sick. 1,354 rows.
3. Dropped 318 rows with no admission count.

### Output: `hospital_admissions_weekly_clean.csv` (15,632 rows)

| column | meaning |
|---|---|
| `state_territory` | 2-letter code, lowercased |
| `week_end` | Saturday ending the CDC epiweek |
| `admits` | **people admitted with confirmed COVID that week** (raw name: `totalconfc19newadm`) |
| `coverage` | fraction of hospitals that actually reported |
| `regime` | which reporting rules were in force |

---

## ⚠️ The week-alignment bug (don't undo this)

Wastewater and NHSN did **not** match up out of the box.

My wastewater aggregation originally ended weeks on **Sunday**. NHSN uses CDC epiweeks ending **Saturday**. One day off → the join matched **zero rows** and returned an empty table **without throwing an error**.

Both sides now use the CDC epiweek (Sunday→Saturday, labelled by the Saturday). **If anyone re-derives `week_end`, keep this convention.**

---

## Files

**Cleaning (my part):**
```
python clean_wastewater.py           # → wastewater_weekly_clean.csv
python clean_hospital_admissions.py  # → hospital_admissions_weekly_clean.csv
```

**Next step (whoever's modeling):**
```
python join_and_build_features.py    # → model_ready_weekly.csv (10,195 rows)
```

Audit trails: `cleaning_log.csv`, `cleaning_log_hospital.csv` — every filter with rows in/out and why.

---

## For the Sources of Bias slide

Two findings better than what's currently on there:

- **Reporting rules changed twice:** mandatory → voluntary (May–Oct 2024) → mandatory. Admissions artificially dip in the middle. We dropped that window.
- **459 weeks have <80% of hospitals reporting.** Those weeks average **43 admissions vs ~450** normally — undercounted, not quiet.

Still true from the original proposal: ~20% of US households are on septic tanks, not sewers, so they're invisible to wastewater monitoring entirely.

---

