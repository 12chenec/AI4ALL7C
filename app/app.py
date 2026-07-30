"""
Streamlit app for the COVID-19 surge prediction model.

Loads the final XGBoost pipeline from final_model/ along with the thresholds
tuned in build_final_pipeline.py, and predicts whether a state is heading into
a COVID admissions surge next week.

Run locally with:  streamlit run app/app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "final_model" / "surge_prediction_pipeline.pkl"
FEATURE_PATH = BASE_DIR / "final_model" / "pipeline_features.txt"
THRESHOLD_PATH = BASE_DIR / "final_model" / "pipeline_thresholds.json"
DATA_PATH = BASE_DIR / "feature_matrix_era2022.csv"

st.set_page_config(page_title="COVID-19 Surge Predictor", layout="centered")


# ---------------------------------------------------------------------------
# Loading (cached so Streamlit doesn't re-read on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = [line.strip() for line in open(FEATURE_PATH) if line.strip()]
    thresholds = json.loads(THRESHOLD_PATH.read_text())
    return model, features, thresholds


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    return df


model, FEATURES, THRESHOLD_INFO = load_model()

# Human-readable labels and help text for each model input.
LABELS = {
    "log10_conc_mean": ("Average wastewater concentration (log10)",
                        "Mean viral concentration across sites this week, log10 scale. Typical range 1.4 to 7.0."),
    "log10_conc_median": ("Median wastewater concentration (log10)",
                          "Median across sites, less affected by one unusual plant."),
    "conc_site_z_mean": ("Site z-score",
                         "How unusual this week is versus each site's own baseline. 0 is typical, positive is elevated."),
    "n_samples": ("Number of samples", "How many individual lab samples back this week's figure."),
    "n_sites": ("Number of sites", "How many distinct treatment plants reported."),
    "pct_nondetect": ("Fraction of non-detect samples",
                      "Share of samples with virus below the detection limit. 0 means every sample detected virus, 1 means none did."),
    "pop_served": ("Population served", "People covered by the reporting wastewater plants."),
    "admits": ("Current COVID hospital admissions", "Confirmed COVID admissions in the state this week."),
    "coverage": ("Hospital reporting coverage", "Fraction of hospitals that reported, from 0 to 1."),
    "admits_per100k": ("Admissions per 100k", "Admissions normalized by the wastewater population served."),
    "log10_conc_lag1": ("Concentration 1 week ago (log10)", "Wastewater signal one week back."),
    "log10_conc_lag2": ("Concentration 2 weeks ago (log10)", "Wastewater signal two weeks back."),
    "log10_conc_lag3": ("Concentration 3 weeks ago (log10)", "Wastewater signal three weeks back."),
    "conc_delta_1w": ("1-week change in concentration", "This week's log10 concentration minus last week's."),
    "conc_roll3": ("3-week rolling average", "Average log10 concentration over the last three weeks."),
    "month": ("Month", "Calendar month, 1 to 12."),
    "epiweek_of_year": ("Epidemiological week", "CDC epiweek number, 1 to 52."),
}

# Reasonable starting values (medians of the 2022+ dataset), so the form is not
# pre-filled with zeros that fall outside the range the model ever saw.
DEFAULTS = {
    "log10_conc_mean": 4.644, "log10_conc_median": 4.678, "conc_site_z_mean": -0.011,
    "n_samples": 28.0, "n_sites": 16.0, "pct_nondetect": 0.0, "pop_served": 2662524.0,
    "admits": 134.0, "coverage": 0.944, "admits_per100k": 5.264,
    "log10_conc_lag1": 4.655, "log10_conc_lag2": 4.661, "log10_conc_lag3": 4.668,
    "conc_delta_1w": -0.008, "conc_roll3": 4.649, "month": 6.0, "epiweek_of_year": 23.0,
}

GROUPS = [
    ("Wastewater signal", ["log10_conc_mean", "log10_conc_median", "conc_site_z_mean",
                           "conc_delta_1w", "conc_roll3"]),
    ("Recent weeks", ["log10_conc_lag1", "log10_conc_lag2", "log10_conc_lag3"]),
    ("Sampling", ["n_samples", "n_sites", "pct_nondetect", "pop_served"]),
    ("Hospital data", ["admits", "coverage", "admits_per100k"]),
    ("Timing", ["month", "epiweek_of_year"]),
]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("COVID-19 Surge Prediction")
st.write(
    "Predicts whether a state is heading into a COVID-19 hospital admissions "
    "surge next week, based on wastewater surveillance and current hospital data. "
    "A surge means admissions rise more than 10% and land above that state's median."
)


# ---------------------------------------------------------------------------
# Sidebar: alert sensitivity
# ---------------------------------------------------------------------------
st.sidebar.header("Alert sensitivity")
st.sidebar.write(
    "The model outputs a probability. This setting controls how high that "
    "probability must be before it is called a surge."
)

SENSITIVITY = {
    "Cautious (fewer false alarms)": "high_precision",
    "Balanced": "balanced",
    "Sensitive (catches the most surges)": "high_recall",
}
choice = st.sidebar.radio("Operating point", list(SENSITIVITY), index=1)
threshold = THRESHOLD_INFO["thresholds"][SENSITIVITY[choice]]

st.sidebar.metric("Decision threshold", f"{threshold:.3f}")
st.sidebar.caption(
    "Thresholds were tuned on a held-out validation window, not on the test set. "
    "The default 0.5 cutoff is a poor fit here because surges are rare, so it "
    "misses most of them."
)

st.sidebar.divider()
st.sidebar.caption(
    "Research and educational use only. Not a clinical or public-health "
    "decision-making tool."
)


# ---------------------------------------------------------------------------
# Input mode
# ---------------------------------------------------------------------------
mode = st.radio(
    "Input",
    ["Load a real state and week", "Enter values manually"],
    horizontal=True,
)

values = dict(DEFAULTS)

if mode == "Load a real state and week":
    data = load_reference_data()
    col_a, col_b = st.columns(2)
    state = col_a.selectbox("State or territory",
                            sorted(data["state_territory"].unique()))
    state_rows = data[data["state_territory"] == state]
    week = col_b.selectbox("Week ending",
                           sorted(state_rows["week_end"].unique(), reverse=True))
    row = state_rows[state_rows["week_end"] == week].iloc[0]
    values = {f: (float(row[f]) if pd.notna(row[f]) else None) for f in FEATURES}
    st.caption(f"Loaded actual surveillance data for {state.upper()}, week ending {week}.")

    with st.expander("View the loaded values"):
        st.dataframe(
            pd.DataFrame({"feature": FEATURES,
                          "value": [values[f] for f in FEATURES]}),
            hide_index=True, width="stretch",
        )
else:
    st.caption("Values start at the dataset median. Adjust whichever matter for your scenario.")
    for group_name, cols in GROUPS:
        st.subheader(group_name)
        for i in range(0, len(cols), 2):
            row_cols = st.columns(2)
            for slot, feat in zip(row_cols, cols[i:i + 2]):
                label, helptext = LABELS[feat]
                if feat == "month":
                    values[feat] = float(slot.selectbox(
                        label, list(range(1, 13)), index=5, help=helptext))
                elif feat == "epiweek_of_year":
                    values[feat] = float(slot.number_input(
                        label, min_value=1, max_value=53,
                        value=int(DEFAULTS[feat]), help=helptext))
                elif feat == "pct_nondetect":
                    values[feat] = slot.slider(
                        label, min_value=0.0, max_value=1.0,
                        value=float(DEFAULTS[feat]), step=0.01, help=helptext)
                elif feat == "coverage":
                    values[feat] = slot.slider(
                        label, min_value=0.0, max_value=1.0,
                        value=float(DEFAULTS[feat]), step=0.01, help=helptext)
                else:
                    values[feat] = slot.number_input(
                        label, value=float(DEFAULTS[feat]), help=helptext)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict", type="primary", width="stretch"):
    input_df = pd.DataFrame([{f: values.get(f) for f in FEATURES}])[FEATURES]

    probability = float(model.predict_proba(input_df)[0][1])
    is_surge = probability >= threshold

    st.subheader("Result")

    if is_surge:
        st.error(
            f"**Surge predicted for next week.** "
            f"Estimated probability {probability:.1%}, at or above the "
            f"{threshold:.3f} threshold for the '{choice.split(' (')[0].lower()}' setting."
        )
    else:
        st.success(
            f"**No surge predicted for next week.** "
            f"Estimated probability {probability:.1%}, below the "
            f"{threshold:.3f} threshold for the '{choice.split(' (')[0].lower()}' setting."
        )

    left, right = st.columns(2)
    left.metric("Surge probability", f"{probability:.1%}")
    right.metric("Threshold in use", f"{threshold:.3f}")
    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "The bar shows the model's estimated probability of a surge, not its "
        "confidence in the answer shown. Changing the alert sensitivity in the "
        "sidebar changes the decision without changing this probability."
    )

    with st.expander("How reliable is this?"):
        st.write(
            "On the held-out 2025 to 2026 test period the model reaches a "
            "ROC-AUC of about 0.87, so it ranks surge weeks well above quiet "
            "weeks. Its absolute precision is limited: surges are rare (about "
            "4% of weeks in that period), so even a good model produces a fair "
            "number of false alarms at sensitive settings. Treat the output as "
            "one signal among several, not a forecast to act on alone."
        )
