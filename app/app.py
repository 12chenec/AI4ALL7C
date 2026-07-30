import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Surge Prediction Dashboard",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "results" / "xgboost_model.pkl"
)

with open(BASE_DIR / "results" / "model_features.txt") as f:
    feature_names = [line.strip() for line in f]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("About This Dashboard")

    st.write(
        """
        This application uses a machine learning model to predict whether
        a COVID-19 surge is likely to occur in the following week.
        """
    )

    st.write(
        """
        The model was trained using wastewater surveillance trends and
        historical COVID-19 outbreak labels.
        """
    )

    st.info(
        """
        Model inputs include wastewater concentration patterns,
        recent trend information, and time-based features.
        """
    )

    st.caption(
        "For research and educational demonstration purposes only."
    )


# -----------------------------
# Header
# -----------------------------
st.title("COVID-19 Surge Prediction Dashboard")

st.markdown(
    """
    This dashboard estimates potential COVID-19 surge risk using
    wastewater surveillance data and machine learning.

    Enter recent wastewater monitoring information below and generate
    a prediction for the following week.
    """
)


st.divider()


# -----------------------------
# Wastewater Monitoring
# -----------------------------
st.subheader("Wastewater Concentration Data")

col1, col2, col3 = st.columns(3)

with col1:
    log10_conc_mean = st.number_input(
        "Average wastewater concentration (log10)",
        value=0.0,
        help="Average measured viral concentration on a logarithmic scale."
    )

with col2:
    log10_conc_median = st.number_input(
        "Median wastewater concentration (log10)",
        value=0.0
    )

with col3:
    conc_site_z_mean = st.number_input(
        "Average concentration trend score",
        value=0.0,
        help="Standardized concentration compared with historical levels."
    )


# -----------------------------
# Sampling Information
# -----------------------------
st.subheader("Wastewater Sampling Information")

col1, col2, col3 = st.columns(3)

with col1:
    n_samples = st.number_input(
        "Number of samples collected",
        min_value=1,
        value=10
    )

with col2:
    n_sites = st.number_input(
        "Number of monitoring sites",
        min_value=1,
        value=1
    )

with col3:
    pct_nondetect = st.number_input(
        "Percentage of non-detect samples",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )


# -----------------------------
# Additional Dataset Features
# -----------------------------
st.subheader("Additional Monitoring Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pop_served = st.number_input(
        "Population represented",
        min_value=0,
        value=100000
    )

with col2:
    admits = st.number_input(
        "Reported hospital admissions",
        min_value=0,
        value=0
    )

with col3:
    coverage = st.number_input(
        "Data coverage score",
        min_value=0.0,
        max_value=1.0,
        value=1.0
    )

with col4:
    admits_per100k = st.number_input(
        "Admissions per 100,000 people",
        min_value=0.0,
        value=0.0
    )


# -----------------------------
# Historical Trends
# -----------------------------
with st.expander("Historical Wastewater Trends"):

    col1, col2, col3 = st.columns(3)

    with col1:
        log10_conc_lag1 = st.number_input(
            "Wastewater concentration 1 week ago",
            value=0.0
        )

    with col2:
        log10_conc_lag2 = st.number_input(
            "Wastewater concentration 2 weeks ago",
            value=0.0
        )

    with col3:
        log10_conc_lag3 = st.number_input(
            "Wastewater concentration 3 weeks ago",
            value=0.0
        )

    col1, col2 = st.columns(2)

    with col1:
        conc_delta_1w = st.number_input(
            "One-week concentration change",
            value=0.0
        )

    with col2:
        conc_roll3 = st.number_input(
            "Three-week rolling average concentration",
            value=0.0
        )


# -----------------------------
# Time Information
# -----------------------------
st.subheader("Time Information")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(
        "Month",
        range(1, 13)
    )

with col2:
    epiweek_of_year = st.number_input(
        "Epidemiological week",
        min_value=1,
        max_value=53,
        value=1
    )


# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button(
    "Generate Prediction",
    use_container_width=True
):

    input_data = {
        "log10_conc_mean": log10_conc_mean,
        "log10_conc_median": log10_conc_median,
        "conc_site_z_mean": conc_site_z_mean,
        "n_samples": n_samples,
        "n_sites": n_sites,
        "pct_nondetect": pct_nondetect,
        "pop_served": pop_served,
        "admits": admits,
        "coverage": coverage,
        "admits_per100k": admits_per100k,
        "log10_conc_lag1": log10_conc_lag1,
        "log10_conc_lag2": log10_conc_lag2,
        "log10_conc_lag3": log10_conc_lag3,
        "conc_delta_1w": conc_delta_1w,
        "conc_roll3": conc_roll3,
        "month": month,
        "epiweek_of_year": epiweek_of_year
    }


    input_df = pd.DataFrame([input_data])

    input_df = input_df[feature_names]


    prediction = model.predict(input_df)[0]

    confidence = model.predict_proba(input_df)[0][1]


    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "A potential COVID-19 surge is predicted for the following week."
        )

    else:

        st.success(
            "No significant COVID-19 surge is predicted for the following week."
        )


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Confidence",
            f"{confidence:.1%}"
        )

    with col2:

        if confidence >= 0.75:
            strength = "High"
        elif confidence >= 0.50:
            strength = "Moderate"
        else:
            strength = "Low"

        st.metric(
            "Prediction Strength",
            strength
        )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Machine learning model trained on wastewater surveillance trends "
    "and historical COVID-19 outbreak labels. "
    "This dashboard is intended for research and educational use."
)