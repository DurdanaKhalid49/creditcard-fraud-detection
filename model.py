import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 Load dataset
df = pd.read_csv("creditcard.csv")  # Update this with your actual dataset path

# 🔹 Drop unnecessary columns (adjust based on your dataset)
df.drop(columns=["V14"], errors="ignore", inplace=True)

# 🔹 Handle categorical features (if any exist)
# df = pd.get_dummies(df, columns=["Category", "PaymentMethod"], drop_first=True)

# 🔹 Define features and target
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target variable

# 🔹 Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Save scaler for later use in the app
joblib.dump(scaler, "scaler.pkl")

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Train model
model = xgb.XGBClassifier(    
    scale_pos_weight=50,  # Increase this based on fraud-to-normal ratio
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)  # Adjust hyperparameters as needed
model.fit(X_train, y_train)

# 🔹 Save model
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")  # Save feature names

print("✅ Fraud Detection Model trained and saved successfully!")
