import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

from DB.data_loader import load_data


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
# 4. FEATURE TYPES
# ==================================================

ALL_CATEGORICAL_FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland",
    "wochentag"
]

categorical_features = [
    feature for feature in selected_features
    if feature in ALL_CATEGORICAL_FEATURES
]

print("\nCategorical features:")
print(categorical_features)


# ==================================================
# 5. PREPROCESSING
# ==================================================
# Gradient Boosting НЕ требует scaling

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==================================================
# 6. MODEL PIPELINE
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )
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

print("\nGradient Boosting training completed.")


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

print("\n===== GRADIENT BOOSTING RESULTS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 11. SAVE MODEL
# ==================================================

joblib.dump(model, "models/gradient_boosting.pkl")

print("Gradient Boosting model saved")