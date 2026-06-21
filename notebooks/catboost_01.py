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

from catboost import CatBoostRegressor
from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. CLEAN DATE (SAFETY)
# ==================================================

for col in df.columns:
    if "datum" in col.lower() or "date" in col.lower():
        df = df.drop(columns=[col])


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
# 4. OUTLIERS
# ==================================================

q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 5. CYCLICAL FEATURES
# ==================================================

df["monat_sin"] = np.sin(2 * np.pi * df["monat"] / 12)
df["monat_cos"] = np.cos(2 * np.pi * df["monat"] / 12)

df["wochentag_sin"] = np.sin(2 * np.pi * df["wochentag"] / 7)
df["wochentag_cos"] = np.cos(2 * np.pi * df["wochentag"] / 7)


# ==================================================
# 6. FEATURES
# ==================================================

TARGET = "preis_euro"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# log transform = stabiler quantile learning
y = np.log1p(y)


# ==================================================
# 7. SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 8. CATEGORICAL FEATURES
# ==================================================

cat_features = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland"
]


# ==================================================
# 9. MODEL FUNCTION
# ==================================================

def train_quantile(alpha):

    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    pred = model.predict(X_test)

    return np.expm1(pred)


# ==================================================
# 10. TRAIN P10 / P50 / P90
# ==================================================

print("\nTraining quantile models...\n")

pred_p10 = train_quantile(0.1)
pred_p50 = train_quantile(0.5)
pred_p90 = train_quantile(0.9)


# ==================================================
# 11. EVALUATION (P50 as main model)
# ==================================================

y_test_real = np.expm1(y_test)
y_pred_real = pred_p50

mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print("\n========== QUANTILE RESULTS ==========")
print(f"MAE (P50): {mae:.2f} €")
print(f"RMSE (P50): {rmse:.2f} €")


# ==================================================
# 12. OUTPUT EXAMPLE
# ==================================================

results = pd.DataFrame({
    "true_price": y_test_real,
    "p10": pred_p10,
    "p50": pred_p50,
    "p90": pred_p90
})

print("\nSample predictions:")
print(results.head(10))