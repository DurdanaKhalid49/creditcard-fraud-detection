import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ✅ 1. Load data
df = pd.read_csv("data/creditcard.csv")

# ✅ 2. Drop unwanted features
columns_to_drop = ['V2','V6','V9','V14','V15','V20','V21','V23','V25','V27','V28']
df.drop(columns=columns_to_drop, inplace=True)

# ✅ 3. Final feature list (store for form input ordering)
final_features = [col for col in df.columns if col != 'Class']

# ✅ 4. Define X and y
X = df[final_features]
y = df['Class']

# ✅ 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 6. Define preprocessing (scaling only)
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), final_features)
    ]
)

# ✅ 7. Define model
model = XGBClassifier(
    scale_pos_weight=50,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ✅ 8. Create pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', model)
])

# ✅ 9. Fit the pipeline
pipeline.fit(X_train, y_train)

# ✅ 10. Save pipeline
import os
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/xgb_fraud_pipeline.joblib")

# ✅ 11. Save feature names (important for form or dashboard)
with open("model/feature_names.txt", "w") as f:
    for col in final_features:
        f.write(col + "\n")

print("✅ Model pipeline saved with 18 features.")
