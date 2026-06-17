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

# Match column names expected by trained models
df = df.rename(columns={
    "Marke": "marke",
    "Modell": "modell",
    "Preis_Euro": "preis_euro"
})

TARGET = "preis_euro"

# ==========================================
# MODELS (ONLY 2-FEATURE MODELS)
# ==========================================

models_info = {

    "Linear Regression": {
        "path": "models/linear_regression.pkl",
        "features": ["marke", "modell"]
    },

    "Linear Regression Basic": {
        "path": "models/linear_regression_basic.pkl",
        "features": ["marke", "modell"]
    },

    "Random Forest": {
        "path": "models/random_forest.pkl",
        "features": ["marke", "modell"]
    },

    "Gradient Boosting": {
        "path": "models/gradient_boosting.pkl",
        "features": ["marke", "modell"]
    }
}

# ==========================================
# EVALUATION
# ==========================================

results = []

for model_name, info in models_info.items():

    print(f"\nEvaluating: {model_name}")

    X = df[info["features"]]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load(info["path"])

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    results.append({
        "Model": model_name,
        "MAE (€)": round(mae, 2),
        "RMSE (€)": round(rmse, 2),
        "R²": round(r2, 4),
        "Error (%)": round(mape, 2),
        "Accuracy (%)": round(100 - mape, 2)
    })

# ==========================================
# RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R²",
    ascending=False
)

print("\n" + "=" * 80)
print("COMPARISON OF MODELS TRAINED WITH 2 FEATURES (marke, modell)")
print("=" * 80)

print(results_df)

results_df.to_csv(
    "fair_model_comparison.csv",
    index=False
)

print("\nResults saved to: fair_model_comparison.csv")