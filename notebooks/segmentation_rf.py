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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. CLEAN STRINGS (IMPORTANT FIX)
# ==================================================

df["marke"] = df["marke"].astype(str).str.strip().str.upper()
df["modell"] = df["modell"].astype(str).str.strip()
df["kraftstoff"] = df["kraftstoff"].astype(str).str.strip()
df["getriebe"] = df["getriebe"].astype(str).str.strip()
df["bundesland"] = df["bundesland"].astype(str).str.strip()


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

df["wochentag"] = df["wochentag"].map(weekday_map)


# ==================================================
# 4. CYCLICAL FEATURES
# ==================================================

df["monat_sin"] = np.sin(2 * np.pi * df["monat"] / 12)
df["monat_cos"] = np.cos(2 * np.pi * df["monat"] / 12)

df["wochentag_sin"] = np.sin(2 * np.pi * df["wochentag"] / 7)
df["wochentag_cos"] = np.cos(2 * np.pi * df["wochentag"] / 7)


# ==================================================
# 5. FEATURES
# ==================================================

TARGET = "preis_euro"

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "hubraum_l",
    "kundenzufriedenheit",
    "bundesland",
    "jahr",
    "monat_sin",
    "monat_cos",
    "wochentag_sin",
    "wochentag_cos"
]

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ==================================================
# 6. COLUMN TYPES
# ==================================================

categorical_features = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland"
]

numeric_features = [
    "hubraum_l",
    "kundenzufriedenheit",
    "jahr",
    "monat_sin",
    "monat_cos",
    "wochentag_sin",
    "wochentag_cos"
]


# ==================================================
# 7. PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# ==================================================
# 8. MODEL
# ==================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])


# ==================================================
# 9. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 10. TRAIN
# ==================================================

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed.")


# ==================================================
# 11. PREDICT
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 12. EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== SEGMENTATION RESULTS ==========")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R²   : {r2:.4f}")
print("=========================================")