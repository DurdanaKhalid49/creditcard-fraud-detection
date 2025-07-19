import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model/xgb_fraud_pipeline.joblib")

# Page configuration
st.set_page_config(page_title="🚨 Credit Card Fraud Detection", layout="wide", page_icon="💳")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("""
This Streamlit app predicts whether a credit card transaction is **fraudulent or safe** using a machine learning model. 
Upload a CSV file or enter transaction details manually to see predictions.
""")

# Sidebar
st.sidebar.header("🔧 Options")
view_option = st.sidebar.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])

# Helper function for prediction
def predict(dataframe):
    prediction = model.predict(dataframe)
    probabilities = model.predict_proba(dataframe)
    return prediction, probabilities

# CSV Upload
if view_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

        if st.button("Predict on Uploaded Data"):
            prediction, prob = predict(data)
            data['Prediction'] = prediction
            data['Fraud Probability'] = prob[:, 1]

            fraud_count = np.sum(prediction == 1)
            safe_count = np.sum(prediction == 0)

            st.success(f"✅ Predictions complete: {fraud_count} Fraud, {safe_count} Safe")

            # Show pie chart
            fig1, ax1 = plt.subplots()
            ax1.pie([fraud_count, safe_count], labels=['Fraud', 'Safe'], autopct='%1.1f%%', colors=['#ff4b4b', '#2ecc71'])
            ax1.axis('equal')
            st.pyplot(fig1)

            st.subheader("Prediction Results")
            st.dataframe(data)

# Manual Input
else:
    st.subheader("Enter Transaction Details Manually")

    input_data = {}
    skip = {2, 6, 9, 14, 15, 20, 21, 23, 25, 27, 28}

    # Group fields in rows of 3 for consistent layout
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            input_data['Time'] = st.number_input("Time", value=0.0,step=1.0 ,format="%.2f")
        with cols[1]:
            input_data['Amount'] = st.number_input("Amount", value=100.0,step=1.0 ,format="%.2f")

    v_features = [i for i in range(1, 29) if i not in skip]

    # Display V1–V28 in groups of 3 per row
    for i in range(0, len(v_features), 3):
        row = st.columns(3)
        for j, v in enumerate(v_features[i:i+3]):
            with row[j]:
                input_data[f'V{v}'] = st.number_input(f'V{v}', value=0.0, format="%.2f")

    if st.button("Predict Transaction"):
        input_df = pd.DataFrame([input_data])
        prediction, prob = predict(input_df)

        result = '🚨 Fraud' if prediction[0] == 1 else '✅ Safe'
        color = '#ff4b4b' if prediction[0] == 1 else '#2ecc71'

        st.markdown(f"<h3 style='color:{color}'>Prediction: {result}</h3>", unsafe_allow_html=True)
        st.write(f"Fraud Probability: {prob[0][1]:.2%}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by [Durdana Khalid](https://github.com/DurdanaKhalid49)")
