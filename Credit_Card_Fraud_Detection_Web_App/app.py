from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/xgb_fraud_pipeline.joblib")

# Load feature names
with open("model/features.txt", "r") as f:
    feature_names = [line.strip() for line in f if line.strip()]

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect values from the form
        input_values = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == "":
                return f"❌ Missing input for feature: {feature}", 400
            input_values.append(float(value))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Transaction is Safe."

        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return f"❌ An error occurred: {str(e)}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
