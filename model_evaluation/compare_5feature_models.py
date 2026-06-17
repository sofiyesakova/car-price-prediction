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

df = pd.read_csv("data/processed/cleaned_data.csv")

df = df.rename(columns={
    "Marke": "marke",
    "Modell": "modell",
    "Preis_Euro": "preis_euro",
    "Kraftstoff": "kraftstoff",
    "Getriebe": "getriebe",
    "Hubraum_L": "hubraum_l"
})

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "hubraum_l"
]

TARGET = "preis_euro"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression Extended": "models/linear_regression_extended.pkl",
    "Random Forest 5 Features": "models/random_forest_5features.pkl"
}

results = []

for model_name, model_path in models.items():
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    error_percent = mean_absolute_percentage_error(y_test, y_pred) * 100

    results.append({
        "Model": model_name,
        "MAE (€)": round(mae, 2),
        "RMSE (€)": round(rmse, 2),
        "R²": round(r2, 4),
        "Error (%)": round(error_percent, 2),
        "Accuracy (%)": round(100 - error_percent, 2)
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R²",
    ascending=False
)

print(results_df)

results_df.to_csv(
    "model_evaluation/comparison_5feature_models.csv",
    index=False
)

print("Saved: model_evaluation/comparison_5feature_models.csv")