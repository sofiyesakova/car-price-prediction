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
# 2. FEATURE SETS
# ==================================================

FEATURE_SETS = {
    "basic": [
        "marke",
        "modell",
    ],

    "extended": [
        "marke",
        "modell",
        "kraftstoff",
        "getriebe",
        "hubraum_l"
    ],

    "all": [
        "marke",
        "modell",
        "verkaufszahl",
        "kraftstoff",
        "getriebe",
        "hubraum_l",
        "kundenzufriedenheit",
        "bundesland",
        "jahr",
        "monat",
        "wochentag"
    ]
}

# Выбираем набор признаков для эксперимента
SELECTED_FEATURE_SET = "basic"

TARGET = "preis_euro"


# ==================================================
# 3. FEATURE SELECTION
# ==================================================

selected_features = FEATURE_SETS[SELECTED_FEATURE_SET]

X = df[selected_features]
y = df[TARGET]

print(f"\nUsing feature set: {SELECTED_FEATURE_SET}")
print(f"Features: {selected_features}")


# ==================================================
# 4. DEFINE FEATURE TYPES
# ==================================================

# Полный список числовых признаков проекта
ALL_NUMERIC_FEATURES = [
    "verkaufszahl",
    "hubraum_l",
    "kundenzufriedenheit",
    "jahr",
    "monat"
]

# Полный список категориальных признаков проекта
ALL_CATEGORICAL_FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland",
    "wochentag"
]

# Автоматически оставляем только те,
# которые присутствуют в текущем наборе фич

numeric_features = [
    feature
    for feature in selected_features
    if feature in ALL_NUMERIC_FEATURES
]

categorical_features = [
    feature
    for feature in selected_features
    if feature in ALL_CATEGORICAL_FEATURES
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ==================================================
# 5. PREPROCESSING
# ==================================================

transformers = []

# Добавляем scaler только если есть числовые признаки
if numeric_features:
    transformers.append(
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    )

# Добавляем encoder только если есть категориальные признаки
if categorical_features:
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


# ==================================================
# 6. MODEL PIPELINE
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==================================================
# 7. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 8. TRAIN MODEL
# ==================================================

model.fit(X_train, y_train)

print("\nModel training completed.")


# ==================================================
# 9. PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 10. EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 11. SAVE MODEL
# ==================================================


joblib.dump(model, "models/linear_regression.pkl")

print("Model saved")