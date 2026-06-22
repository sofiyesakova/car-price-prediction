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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from catboost import CatBoostClassifier
from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print("Dataset shape:", df.shape)

df = df.copy()


# ==================================================
# 2. DROP DATE FEATURES
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
# 4. FEATURE ENGINEERING (STRONG + SIMPLE)
# ==================================================

CURRENT_YEAR = 2026

if "baujahr" in df.columns:
    df["age"] = CURRENT_YEAR - df["baujahr"]

if "kilometerstand" in df.columns and "age" in df.columns:
    df["km_per_year"] = df["kilometerstand"] / (df["age"] + 1)

# interaction feature
if "marke" in df.columns and "modell" in df.columns:
    df["marke_modell"] = (
        df["marke"].astype(str) + "_" +
        df["modell"].astype(str).str[:10]
    )


# extra useful interactions
if "hubraum_l" in df.columns:
    df["power_proxy"] = df["hubraum_l"] * df.get("verkaufszahl", 1)


# ==================================================
# 5. CLEAN OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]

q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 6. TARGET (2 CLASSES: PREMIUM vs REST)
# ==================================================

PREMIUM_THRESHOLD = 79716  # Hochpreis boundary

df["target"] = (df["preis_euro"] > PREMIUM_THRESHOLD).astype(int)

print("\nClass distribution:")
print(df["target"].value_counts())


# ==================================================
# 7. FEATURES / TARGET SPLIT
# ==================================================

X = df.drop(columns=["preis_euro", "target"])
y = df["target"]


# ==================================================
# 8. TRAIN / TEST SPLIT
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
# 10. MODEL (FIXED + STABLE)
# ==================================================

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=7,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
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
# 11. EVALUATION
# ==================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n========== RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==================================================
# 12. FEATURE IMPORTANCE
# ==================================================

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop features:")
print(fi.head(15))