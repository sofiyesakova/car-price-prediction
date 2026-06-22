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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
df = df.copy()

print("Dataset shape:", df.shape)


# ==================================================
# 2. FEATURE ENGINEERING (safe)
# ==================================================

CURRENT_YEAR = 2026

if "baujahr" in df.columns:
    df["age"] = CURRENT_YEAR - df["baujahr"]

if "kilometerstand" in df.columns and "age" in df.columns:
    df["km_per_year"] = df["kilometerstand"] / (df["age"] + 1)


# ==================================================
# 3. OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]
q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 4. FEATURE SET
# ==================================================

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland",
    "wochentag",
    "hubraum_l",
    "verkaufszahl",
    "kundenzufriedenheit",
    "age",
    "km_per_year"
]

FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]


# ==================================================
# 5. CATEGORICAL FEATURES
# ==================================================

cat_features = [
    c for c in ["marke", "modell", "kraftstoff", "getriebe", "bundesland", "wochentag"]
    if c in FEATURES
]


# ==================================================
# 6. PREPROCESSOR (YOUR WORKING VERSION)
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_features
        )
    ],
    remainder="passthrough"
)


# ==================================================
# ==================================================
# 🔥 7A. BINARY CLASSIFICATION
# ==================================================
# ==================================================

df_bin = df.copy()

df_bin["target"] = (df_bin["preis_euro"] > 79716).astype(int)

X_bin = df_bin[FEATURES]
y_bin = df_bin["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_bin
)

model_bin = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

model_bin.fit(X_train, y_train)

pred_bin = model_bin.predict(X_test)

print("\n========== RF 2-CLASS ==========")
print("Accuracy:", accuracy_score(y_test, pred_bin))
print(classification_report(y_test, pred_bin))


# ==================================================
# ==================================================
# 🔥 7B. MULTICLASS (3 CLASS)
# ==================================================
# ==================================================

df_multi = df.copy()

def make_segment(price):
    if price < 48545:
        return 0
    elif price < 79716:
        return 1
    else:
        return 2


df_multi["target"] = df_multi["preis_euro"].apply(make_segment)

X_multi = df_multi[FEATURES]
y_multi = df_multi["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi,
    test_size=0.2,
    random_state=42,
    stratify=y_multi
)

model_multi = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

model_multi.fit(X_train, y_train)

pred_multi = model_multi.predict(X_test)

print("\n========== RF 3-CLASS ==========")
print("Accuracy:", accuracy_score(y_test, pred_multi))
print(classification_report(y_test, pred_multi))