import sys
import os

# добавляем корень проекта в PYTHONPATH
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print(f"Data shape: {df.shape}")


# ==================================================
# 2. FEATURE SET
# ==================================================

FEATURE_SETS = {
    "basic": ["marke", "modell"],
    "extended": ["marke", "modell", "kraftstoff", "getriebe", "hubraum_l"],
    "all": [
        "marke", "modell", "verkaufszahl",
        "kraftstoff", "getriebe", "hubraum_l",
        "kundenzufriedenheit", "bundesland",
        "jahr", "monat", "wochentag"
    ]
}

SELECTED_FEATURE_SET = "all"
TARGET = "preis_euro"

selected_features = FEATURE_SETS[SELECTED_FEATURE_SET]

X = df[selected_features]
y = df[TARGET]


print(f"\nUsing feature set: {SELECTED_FEATURE_SET}")
print(f"Features: {selected_features}")


# ==================================================
# 3. COLUMN TYPES
# ==================================================

numeric_features = [
    col for col in selected_features
    if col in ["verkaufszahl", "hubraum_l", "kundenzufriedenheit", "jahr", "monat"]
]

categorical_features = [
    col for col in selected_features
    if col in ["marke", "modell", "kraftstoff", "getriebe", "bundesland", "wochentag"]
]


print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ==================================================
# 4. PREPROCESSOR
# ==================================================

transformers = []

if numeric_features:
    transformers.append(
        ("num", StandardScaler(), numeric_features)
    )

if categorical_features:
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)


# ==================================================
# 5. MODEL PIPELINE
# ==================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# ==================================================
# 6. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# 7. TRAIN
# ==================================================

model.fit(X_train, y_train)
print("\nModel training completed.")


# ==================================================
# 8. PREDICT
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 9. EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 10. SAVE
# ==================================================

import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "linear_regression_all.pkl"
)

joblib.dump(model, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")
