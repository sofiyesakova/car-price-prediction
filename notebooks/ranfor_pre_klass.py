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

df = load_data().copy()

print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. REMOVE EXTREME OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]

q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# 3. FEATURES (ONLY YOUR FEATURES)
# ==================================================

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland",
    "wochentag",
    "verkaufszahl",
    "hubraum_l",
    "jahr",
    "monat"
]

FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]


# ==================================================
# 4. CATEGORICAL FEATURES
# ==================================================

CATEGORICAL_FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "bundesland",
    "wochentag"
]

CATEGORICAL_FEATURES = [
    f for f in CATEGORICAL_FEATURES
    if f in FEATURES
]


# ==================================================
# 5. PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ],
    remainder="passthrough"
)


# ==================================================
# ==================================================
# 6A. BINARY CLASSIFICATION
# Hochpreis vs Rest
# ==================================================
# ==================================================

df_bin = df.copy()

df_bin["target"] = (
    df_bin["preis_euro"] >= 79716
).astype(int)

X_bin = df_bin[FEATURES]
y_bin = df_bin["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_bin,
    y_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_bin
)

rf_bin = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)

rf_bin.fit(X_train, y_train)

y_pred = rf_bin.predict(X_test)

print("\n========== RF 2-CLASS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ==================================================
# ==================================================
# 6B. MULTI-CLASS CLASSIFICATION
# Niedrigpreis / Mittelklasse / Hochpreis
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


df_multi["target"] = (
    df_multi["preis_euro"]
    .apply(make_segment)
)

X_multi = df_multi[FEATURES]
y_multi = df_multi["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42,
    stratify=y_multi
)

rf_multi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
        )
    ]
)

rf_multi.fit(X_train, y_train)

y_pred = rf_multi.predict(X_test)

print("\n========== RF 3-CLASS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))