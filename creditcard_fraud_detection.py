from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 🔹 Load saved model, scaler, and feature names
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

@app.route('/')
def home():
    print("Feature Names:", feature_names)  # Debugging Line
    return render_template("index.html", feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Get user input from form
        input_data = [float(request.form[feature]) for feature in feature_names]

        # 🔹 Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # 🔹 Scale input data
        input_scaled = scaler.transform(input_df)

        # 🔹 Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Fraud probability

        result = "Fraudulent Transaction!" if prediction == 1 else "Legitimate Transaction!"
        confidence = f"{probability:.2%}" if prediction == 1 else f"{(1-probability):.2%}"

        return render_template("results.html", result=result, confidence=confidence)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
