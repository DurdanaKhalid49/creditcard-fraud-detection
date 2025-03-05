import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 🔹 Load saved model, scaler, and feature names
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# 🔹 Streamlit UI
st.title("💳 Fraud Detection Web App")
st.markdown("Enter transaction details to check if it's fraudulent or not.")

# 🔹 User Input Form
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# 🔹 Convert input into DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# 🔹 Scale the input data
input_scaled = scaler.transform(input_df)

# 🔹 Prediction
if st.button("Detect Fraud 🚀"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Fraud probability

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Transaction is Legitimate! (Confidence: {(1-probability):.2%})")
