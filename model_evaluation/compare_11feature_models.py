import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/cleaned_data.csv")

df = df.rename(columns={
    "Marke": "marke",
    "Modell": "modell",
    "Preis_Euro": "preis_euro",
    "Verkaufszahl": "verkaufszahl",
    "Kraftstoff": "kraftstoff",
    "Getriebe": "getriebe",
    "Hubraum_L": "hubraum_l",
    "Bundesland": "bundesland",
    "Kundenzufriedenheit": "kundenzufriedenheit",
    "Jahr": "jahr",
    "Monat": "monat",
    "Wochentag": "wochentag"
})

# ==========================================
# FEATURES
# ==========================================

FEATURES = [
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

TARGET = "preis_euro"

X = df[FEATURES]
y = df[TARGET]

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODELS
# ==========================================

models = {
    "Linear Regression All":
        "models/linear_regression_all.pkl",

    "Random Forest 11 Features":
        "models/random_forest_11features.pkl"
}

# ==========================================
# EVALUATION
# ==========================================

results = []

for model_name, model_path in models.items():

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    error_percent = (
        mean_absolute_percentage_error(
            y_test,
            y_pred
        ) * 100
    )

    results.append({
        "Model": model_name,
        "MAE (€)": round(mae, 2),
        "RMSE (€)": round(rmse, 2),
        "R²": round(r2, 4),
        "Error (%)": round(error_percent, 2),
        "Accuracy (%)": round(100 - error_percent, 2)
    })

# ==========================================
# RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R²",
    ascending=False
)

print("\n===== 11 FEATURE MODEL COMPARISON =====")
print(results_df)

results_df.to_csv(
    "model_evaluation/comparison_11feature_models.csv",
    index=False
)

print(
    "\nSaved: model_evaluation/comparison_11feature_models.csv"
)