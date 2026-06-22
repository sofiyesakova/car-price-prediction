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
# 2. CLEAN DATE COLUMNS
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
# 4. FEATURE ENGINEERING (IMPORTANT UPGRADE)
# ==================================================

current_year = 2026

# age feature (VERY IMPORTANT for car prices)
if "baujahr" in df.columns:
    df["age"] = current_year - df["baujahr"]

# mileage per year (VERY strong feature if km exists)
if "kilometerstand" in df.columns and "age" in df.columns:
    df["km_per_year"] = df["kilometerstand"] / (df["age"] + 1)

# interaction feature (VERY useful for CatBoost)
df["marke_modell"] = df["marke"].astype(str) + "_" + df["modell"].astype(str)


# ==================================================
# 5. OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]  # safety filter
q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 6. CYCLICAL FEATURES
# ==================================================

if "monat" in df.columns:
    df["monat_sin"] = np.sin(2 * np.pi * df["monat"] / 12)
    df["monat_cos"] = np.cos(2 * np.pi * df["monat"] / 12)

if "wochentag" in df.columns:
    df["wochentag_sin"] = np.sin(2 * np.pi * df["wochentag"] / 7)
    df["wochentag_cos"] = np.cos(2 * np.pi * df["wochentag"] / 7)


# ==================================================
# 7. TARGET
# ==================================================

TARGET = "preis_euro"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ❗ IMPORTANT: try WITHOUT log first (more stable MAE)
# y = np.log1p(y)


# ==================================================
# 8. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ==================================================
# 9. CATEGORICAL FEATURES (UPDATED)
# ==================================================

cat_features = []

for col in ["marke", "modell", "kraftstoff", "getriebe", "bundesland", "marke_modell"]:
    if col in X.columns:
        cat_features.append(col)


# ==================================================
# 10. MODEL CONFIG (IMPROVED)
# ==================================================

def train_quantile(alpha):

    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=2000,
        learning_rate=0.03,
        depth=6,              # IMPORTANT: lower depth = better generalization
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        od_type="Iter",
        od_wait=100          # early stopping
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    pred = model.predict(X_test)

    return pred


# ==================================================
# 11. TRAIN MODELS
# ==================================================

print("\nTraining quantile models...\n")

pred_p10 = train_quantile(0.1)
pred_p50 = train_quantile(0.5)
pred_p90 = train_quantile(0.9)


# ==================================================
# 12. EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, pred_p50)
rmse = np.sqrt(mean_squared_error(y_test, pred_p50))

print("\n========== QUANTILE RESULTS ==========")
print(f"MAE (P50): {mae:.2f} €")
print(f"RMSE (P50): {rmse:.2f} €")


# ==================================================
# 13. RESULTS
# ==================================================

results = pd.DataFrame({
    "true_price": y_test.values,
    "p10": pred_p10,
    "p50": pred_p50,
    "p90": pred_p90
})

print("\nSample predictions:")
print(results.head(10))