import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Surge Predictor",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 COVID-19 Wastewater Surge Prediction")

st.write(
    """
Enter wastewater surveillance measurements below to estimate whether
COVID-19 cases are likely to surge next week.
"""
)

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "results" / "xgboost_model.pkl")

with open(BASE_DIR / "results" / "model_features.txt") as f:
    feature_names = [line.strip() for line in f]

# -----------------------------
# User Inputs
# -----------------------------
st.header("Input Features")

user_input = {}

for feature in feature_names:
    user_input[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Prediction")

    if prediction == 1:
        st.error("⚠️ High Risk of COVID-19 Surge")
    else:
        st.success("✅ Low Risk of COVID-19 Surge")

    st.write(f"Prediction confidence: **{probability:.1%}**")