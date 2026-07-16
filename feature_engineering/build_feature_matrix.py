"""
Member 3: Feature Engineering - Anika
AI4ALL Group 7C

Takes Amy's model_ready_weekly.csv / model_ready_era2022.csv (already merged +
partially feature-engineered by Amy/Christal) and turns them into a clean,
modeling-safe feature matrix:

  1. Adds temporal features (month, season) and geographic features (US region)
     from week_end and state_territory.
  2. One-hot encodes categorical fields (regime, season, region).
  3. Splits every "future" / leakage column out into a separate targets table,
     so a modeler can never accidentally train on the answer.
  4. Writes a column dictionary so everyone knows what's a feature vs. a target.

Run after join_and_build_features.py:
    python build_feature_matrix.py
"""
import pandas as pd
import numpy as np
import os

IN_DIR = "."   # run from repo root, or edit this path
OUT_DIR = "."

# ---------------------------------------------------------------------------
# Columns that are derived from *next week* (the thing we're predicting).
# These must NEVER be used as model inputs - only as labels.
# ---------------------------------------------------------------------------
LEAKAGE_COLS = [
    "admits_next_week", "y_reg_next_admits", "pct_change_next",
    "y_surge_next_week", "next_week_end",
]

# Identifier columns - useful for joining/inspecting, not features themselves.
ID_COLS = ["state_territory", "week_end"]

# Exact duplicate of `admits` (see review doc) - drop to avoid double-counting.
REDUNDANT_COLS = ["admits_this_week"]

US_REGION = {
    # Northeast
    "ct": "northeast", "me": "northeast", "ma": "northeast", "nh": "northeast",
    "ri": "northeast", "vt": "northeast", "nj": "northeast", "ny": "northeast", "pa": "northeast",
    # Midwest
    "il": "midwest", "in": "midwest", "mi": "midwest", "oh": "midwest", "wi": "midwest",
    "ia": "midwest", "ks": "midwest", "mn": "midwest", "mo": "midwest", "ne": "midwest",
    "nd": "midwest", "sd": "midwest",
    # South
    "de": "south", "fl": "south", "ga": "south", "md": "south", "nc": "south", "sc": "south",
    "va": "south", "dc": "south", "wv": "south", "al": "south", "ky": "south", "ms": "south",
    "tn": "south", "ar": "south", "la": "south", "ok": "south", "tx": "south",
    # West
    "az": "west", "co": "west", "id": "west", "mt": "west", "nv": "west", "nm": "west",
    "ut": "west", "wy": "west", "ak": "west", "ca": "west", "hi": "west", "or": "west", "wa": "west",
}


def add_temporal_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_end"] = pd.to_datetime(df["week_end"])

    # --- temporal ---
    df["month"] = df["week_end"].dt.month
    df["epiweek_of_year"] = df["week_end"].dt.isocalendar().week.astype(int)
    df["season"] = pd.cut(
        df["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "fall", "winter"],
        ordered=False,
    )
    # December wraps to "winter" via the two bin edges above; fix Dec (12) explicitly
    df.loc[df["month"] == 12, "season"] = "winter"

    # --- geographic ---
    df["region"] = df["state_territory"].str.lower().map(US_REGION).fillna("territory")

    return df


def build_matrix(path_in: str, path_features_out: str, path_targets_out: str) -> pd.DataFrame:
    df = pd.read_csv(path_in)
    df = add_temporal_geo_features(df)
    df = df.drop(columns=[c for c in REDUNDANT_COLS if c in df.columns])

    # one-hot encode categoricals
    cat_cols = [c for c in ["regime", "season", "region"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    present_leak = [c for c in LEAKAGE_COLS if c in df.columns]
    targets = df[ID_COLS + present_leak].copy()
    features = df.drop(columns=present_leak)

    features.to_csv(path_features_out, index=False)
    targets.to_csv(path_targets_out, index=False)

    print(f"{path_in}")
    print(f"  -> {path_features_out}  {features.shape}")
    print(f"  -> {path_targets_out}   {targets.shape}")
    return features


if __name__ == "__main__":
    feat_weekly = build_matrix(
        os.path.join(IN_DIR, "model_ready_weekly.csv"),
        os.path.join(OUT_DIR, "feature_matrix_weekly.csv"),
        os.path.join(OUT_DIR, "targets_weekly.csv"),
    )
    feat_era = build_matrix(
        os.path.join(IN_DIR, "model_ready_era2022.csv"),
        os.path.join(OUT_DIR, "feature_matrix_era2022.csv"),
        os.path.join(OUT_DIR, "targets_era2022.csv"),
    )

    print("\nFinal feature columns (era2022 matrix):")
    for c in feat_era.columns:
        print(f"  - {c}")

    # quick NaN check on the lag features (expected for first weeks of each state)
    nan_report = feat_era[[c for c in feat_era.columns if "lag" in c]].isna().sum()
    print("\nNaNs in lag features (expected for each state's first 1-2 weeks):")
    print(nan_report)
