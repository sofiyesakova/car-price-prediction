import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from db.data_loader import load_data


# ==================================================
# LOAD DATA
# ==================================================

df = load_data().copy()

print(f"Dataset shape: {df.shape}")


# ==================================================
# OUTLIERS
# ==================================================

df = df[df["preis_euro"] > 1000]

q99 = df["preis_euro"].quantile(0.99)
df = df[df["preis_euro"] <= q99]


# ==================================================
# FEATURES
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

FEATURES = [
    f for f in FEATURES
    if f in df.columns
]

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
# TARGET
# ==================================================

df["target"] = (
    df["preis_euro"] >= 75716
).astype(int)

X = df[FEATURES]
y = df["target"]


# ==================================================
# PREPROCESSING
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
# TRAIN / TEST
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# MODEL
# ==================================================

model = Pipeline(
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

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n========== RF 2-CLASS ==========")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


# ==================================================
# SAVE MODEL
# ==================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "random_forest_fahrzeugpreiskategorie_2klassen.pkl"
)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved:\n{MODEL_PATH}")