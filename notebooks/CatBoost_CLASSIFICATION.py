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
from sklearn.metrics import classification_report, accuracy_score

from catboost import CatBoostClassifier
from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print("Dataset shape:", df.shape)

df = df.copy()


# ==================================================
# 2. DROP DATE COLUMNS
# ==================================================

df = df.drop(
    columns=[c for c in df.columns if "datum" in c.lower() or "date" in c.lower()],
    errors="ignore"
)


# ==================================================
# 3. WEEKDAY ENCODING
# ==================================================

weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4,
    "Saturday": 5, "Sunday": 6
}

if "wochentag" in df.columns:
    df["wochentag"] = df["wochentag"].map(weekday_map)


# ==================================================
# 4. FEATURE ENGINEERING (SIMPLIFIED BUT STRONG)
# ==================================================

CURRENT_YEAR = 2026

if "baujahr" in df.columns:
    df["age"] = CURRENT_YEAR - df["baujahr"]

if "kilometerstand" in df.columns and "age" in df.columns:
    df["km_per_year"] = df["kilometerstand"] / (df["age"] + 1)

# interaction (important but stable)
if "marke" in df.columns and "modell" in df.columns:
    df["marke_modell"] = (
        df["marke"].astype(str) + "_" + df["modell"].astype(str).str[:10]
    )


# ==================================================
# 5. OUTLIERS CLEANING
# ==================================================

df = df[df["preis_euro"] > 1000]

q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 6. TARGET CREATION (PRICE SEGMENTS)
# ==================================================

def make_segment(price):
    if price < 48545:
        return 0   # Niedrigpreis
    elif price < 79716:
        return 1   # Mittelklasse
    else:
        return 2   # Hochpreis


df["target"] = df["preis_euro"].apply(make_segment)


# ==================================================
# 7. FEATURES / TARGET SPLIT
# ==================================================

X = df.drop(columns=["preis_euro", "target"])
y = df["target"]


# ==================================================
# 8. TRAIN / TEST SPLIT (IMPORTANT: STRATIFY)
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 9. CATEGORICAL FEATURES
# ==================================================

cat_features = [
    col for col in [
        "marke",
        "modell",
        "kraftstoff",
        "getriebe",
        "bundesland",
        "marke_modell"
    ] if col in X.columns
]


# ==================================================
# 10. CLASS BALANCING (IMPORTANT FIX)
# ==================================================

class_counts = y_train.value_counts().sort_index()
total = len(y_train)

class_weights = [
    total / (len(class_counts) * class_counts[i])
    for i in range(len(class_counts))
]

print("Class weights:", class_weights)


# ==================================================
# 11. CATBOOST MODEL (FIXED VERSION)
# ==================================================

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    class_weights=class_weights,
    random_seed=42,
    verbose=200,
    od_type="Iter",
    od_wait=300
)


model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)


# ==================================================
# 12. EVALUATION
# ==================================================

y_pred = model.predict(X_test)

print("\n========== RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==================================================
# 13. FEATURE IMPORTANCE
# ==================================================

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop features:")
print(fi.head(15))