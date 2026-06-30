import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. BASIC CLEANING
# ==================================================

df = df.copy()

for col in df.columns:
    if "datum" in col.lower() or "date" in col.lower():
        df.drop(columns=[col], inplace=True)


# ==================================================
# 3. WEEKDAY FIX
# ==================================================

weekday_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

if "wochentag" in df.columns:
    df["wochentag"] = df["wochentag"].map(weekday_map)


# ==================================================
# 4. OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]
df = df[df["preis_euro"] <= df["preis_euro"].quantile(0.99)]


# ==================================================
# 5. TARGET
# ==================================================

X = df.drop(columns=["preis_euro"])
y = df["preis_euro"]


# ==================================================
# 6. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 7. CATEGORICAL + NUMERIC FEATURES
# ==================================================

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]


# ==================================================
# 8. PREPROCESSING (IMPORTANT for XGBoost)
# ==================================================

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)


# ==================================================
# 9. MODEL
# ==================================================

xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)


model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb)
])


# ==================================================
# 10. TRAIN
# ==================================================

print("\nTraining XGBoost model...\n")

model.fit(X_train, y_train)


# ==================================================
# 11. PREDICT
# ==================================================

pred = model.predict(X_test)


# ==================================================
# 12. EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)



print("\n========== XGBOOST RESULTS ==========")
print(f"MAE: {mae:.2f} €")
print(f"RMSE: {rmse:.2f} €")
print(f"R²: {r2:.4f}")