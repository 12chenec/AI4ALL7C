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

WW_FILE = BASE_DIR / "data_cleaning" / "wastewater_weekly_clean.csv"
HOSP_FILE = BASE_DIR / "data_cleaning" / "hospital_admissions_weekly_clean.csv"

ww = pd.read_csv(WW_FILE)
hosp = pd.read_csv(HOSP_FILE)

ww["week_end"] = pd.to_datetime(ww["week_end"])
hosp["week_end"] = pd.to_datetime(hosp["week_end"])

states = sorted(ww["state_territory"].unique())


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
# Location Selection
# -----------------------------

st.subheader("Select Location")

state = st.selectbox(
    "Choose a state",
    states
)


state_ww = (
    ww[ww["state_territory"] == state]
    .sort_values("week_end")
)

latest_data = state_ww.tail(1).copy()

# wastewater lag features
for L in [1,2,3]:
    latest_data[f"log10_conc_lag{L}"] = state_ww["log10_conc_mean"].shift(L).iloc[-1]

latest_data["conc_delta_1w"] = (
    latest_data["log10_conc_mean"].iloc[0]
    -
    latest_data["log10_conc_lag1"].iloc[0]
)

latest_data["conc_roll3"] = (
    state_ww["log10_conc_mean"]
    .tail(3)
    .mean()
)

# time features
latest_data["month"] = latest_data["week_end"].dt.month.iloc[0]
latest_data["epiweek_of_year"] = (
    latest_data["week_end"].dt.isocalendar().week.iloc[0]
)


# add latest available hospital info
latest_hosp = (
    hosp[hosp["state_territory"] == state]
    .sort_values("week_end")
    .tail(1)
)

latest_data["admits"] = latest_hosp["admits"].iloc[0]
latest_data["coverage"] = latest_hosp["coverage"].iloc[0]


latest_data["admits_per100k"] = (
    latest_data["admits"].iloc[0] /
    1e5
)


st.write(
    f"Using latest available data: {latest_data['week_end'].iloc[0]}"
)


# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button(
    "Generate Prediction",
    use_container_width=True
):

    input_df = latest_data[feature_names]

    prediction = model.predict(input_df)[0]

    confidence = model.predict_proba(input_df)[0][1]


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
        if prediction == 1:
            confidence = model.predict_proba(input_df)[0][1]
        else:
            confidence = model.predict_proba(input_df)[0][0]

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