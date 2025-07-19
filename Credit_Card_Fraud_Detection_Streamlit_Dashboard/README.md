# 💳 Credit Card Fraud Detection Dashboard

A Streamlit web app that predicts whether a credit card transaction is **fraudulent** or **safe** using a trained **XGBoost Machine Learning model**. This project is part of my data science portfolio and showcases model deployment and interactive user input capabilities.

![Streamlit Dashboard Screenshot](https://raw.githubusercontent.com/DurdanaKhalid49/creditcard-fraud-detection/main/preview.png) <!-- Add a screenshot if available -->

---

## 🚀 Live Demo

🔗 [Launch the App](https://your-app-url.up.railway.app)

---

## 📌 Features

- 🧠 Fraud prediction using a trained XGBoost pipeline
- 📥 Upload CSV for bulk prediction
- 🧾 Manual input mode for single transaction analysis
- 📊 Fraud vs Safe Pie Chart Visualization
- ✅ Confidence scores for predictions
- 💡 Clean and user-friendly layout

---

## 🧠 Model Info

- **Algorithm:** XGBoost Classifier
- **Pipeline:** Includes preprocessing, feature scaling, and model
- **Input Features:** `Time`, `V1–V28` (excluding dropped components), `Amount`

---

## 🖥️ Usage

### 🔹 Option 1: Upload CSV

Upload a file containing transaction data, and the app will predict fraud probability for each row.

### 🔹 Option 2: Manual Input

Enter transaction values manually for real-time prediction.

---

## 🛠️ Tech Stack

- `Python 3.10`
- `Streamlit`
- `Pandas`, `NumPy`
- `Matplotlib`, `Seaborn`
- `XGBoost`
- `joblib`

---

## 📁 Project Structure

Credit_Card_Fraud_Detection_Streamlit_Dashboard/
│
├── app.py # Streamlit app
├── requirements.txt # Required packages
├── model/
│ └── xgb_fraud_pipeline.joblib # Saved ML pipeline
├── Procfile # For deployment (Railway)
└── README.md


---

## 🧪 Run Locally

```bash
git clone https://github.com/DurdanaKhalid49/creditcard-fraud-detection.git
cd creditcard-fraud-detection/Credit_Card_Fraud_Detection_Streamlit_Dashboard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
```
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
📸 Screenshots
Manual Entry Prediction	CSV Upload Results

📬 Contact
Made with ❤️ by Durdana Khalid
🔗 GitHub
📫 For collaboration or freelance inquiries: [Email or LinkedIn]

📄 License
This project is open-source and available under the MIT License.
