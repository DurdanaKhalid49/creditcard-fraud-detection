# 💳 Credit Card Fraud Detection – Flask Web App

This project is a Flask-based web application that allows users to input transaction details and get a prediction on whether the transaction is fraudulent or safe using a trained XGBoost model.

> ⚡ Built with a production-ready machine learning pipeline and tested on real-world credit card fraud data.

---

## 📌 Project Highlights

- ✅ Built with **Flask** for backend logic
- ✅ Pre-trained **XGBoost** model wrapped in a **sklearn Pipeline**
- ✅ End-to-end fraud detection with **19 cleaned features**
- ✅ Fully tested on fraud + non-fraud inputs
- ✅ User-friendly and responsive frontend (HTML + CSS)
- ✅ Ready for deployment (Railway, Render, etc.)

---

## 🚀 Demo (Optional GIF/Screenshot)
*Add a screenshot or GIF of the working app here if available*

---

## 🧠 How It Works

1. The user enters transaction details via the form.
2. The app sends the inputs to the Flask backend.
3. The saved ML pipeline processes the input and makes a prediction.
4. The result (Fraud or Safe) is displayed on the page.

---

## 📂 Project Structure
```
Credit_Card_Fraud_Detection_Flask_Web_App/
│
├── app.py # Flask application
├── model/
│ └── xgb_fraud_pipeline.joblib # Trained model with preprocessing
├── templates/
│ └── index.html # Frontend form
├── static/
│ └── style.css # Custom CSS styling
├── requirements.txt # All dependencies
└── README.md # Project documentation
```

## 🛠 Technologies Used

- Python 3.10
- Flask
- Scikit-learn
- XGBoost
- HTML/CSS (Jinja2 templating)
- NumPy

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```
   git clone https://github.com/your-username/Credit_Card_Fraud_Detection_Flask_Web_App.git
   cd Credit_Card_Fraud_Detection_Flask_Web_App
    ```
Create and Activate Virtual Environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
Install Requirements
```
pip install -r requirements.txt
```
Run the App

```
python app.py
Open in Browser
Navigate to http://127.0.0.1:5000 in your web browser.
```
✅ Sample Inputs for Testing
You can use both fraud and non-fraud sample inputs to test the model. A few samples are included in the app for reference.

🌐 Deployment Ready
Compatible with platforms like Railway, Render, and Heroku

No dataset needed at runtime — only the saved model is required

🙋‍♀️ Author
Durdana Khalid
Beginner Data Scientist | Passionate about deploying real-world ML solutions
LinkedIn | Portfolio

📜 License
This project is licensed under the MIT License.
