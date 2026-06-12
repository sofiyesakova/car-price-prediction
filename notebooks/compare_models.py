import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()

print(f"Data shape: {df.shape}")


# ==================================================
# 2. FEATURES (должны совпадать с обучением моделей)
# ==================================================

FEATURES = [
    "marke",
    "modell"
]

TARGET = "preis_euro"

X = df[FEATURES]
y = df[TARGET]


# ==================================================
# 3. TRAIN / TEST SPLIT (ОДИНАКОВЫЙ ДЛЯ ВСЕХ МОДЕЛЕЙ)
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 4. LOAD MODELS
# ==================================================

linear_model = joblib.load("models/linear_regression.pkl")
rf_model = joblib.load("models/random_forest.pkl")
gb_model = joblib.load("models/gradient_boosting.pkl")


models = {
    "Linear Regression": linear_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}


# ==================================================
# 5. EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==================================================
# 6. RUN COMPARISON
# ==================================================

results = []

for name, model in models.items():
    res = evaluate_model(name, model, X_test, y_test)
    results.append(res)


results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.sort_values(by="R2", ascending=False))


# ==================================================
# 7. BEST MODEL
# ==================================================

best_model = results_df.sort_values(by="R2", ascending=False).iloc[0]

print("\n===== BEST MODEL =====")
print(best_model)