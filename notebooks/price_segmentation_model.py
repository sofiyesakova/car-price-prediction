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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from db.data_loader import load_data


# ==================================================
# 1. LOAD DATA
# ==================================================

df = load_data()
print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. CLEAN DATA
# ==================================================

df["marke"] = df["marke"].astype(str).str.strip().str.upper()
df["modell"] = df["modell"].astype(str).str.strip()
df["kraftstoff"] = df["kraftstoff"].astype(str).str.strip()
df["getriebe"] = df["getriebe"].astype(str).str.strip()
df["bundesland"] = df["bundesland"].astype(str).str.strip()


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
# 4. CYCLICAL FEATURES
# ==================================================

df["monat_sin"] = np.sin(2 * np.pi * df["monat"] / 12)
df["monat_cos"] = np.cos(2 * np.pi * df["monat"] / 12)

df["wochentag_sin"] = np.sin(2 * np.pi * df["wochentag"] / 7)
df["wochentag_cos"] = np.cos(2 * np.pi * df["wochentag"] / 7)


# ==================================================
# 5. PRICE SEGMENTS (TARGET)
# ==================================================

df = df[df["preis_euro"].notna()]

df["price_segment"] = pd.qcut(
    df["preis_euro"],
    q=4,
    labels=[0, 1, 2, 3]
)

print("\nPrice segment distribution:")
print(df["price_segment"].value_counts().sort_index())


# ==================================================
# 6. FEATURES
# ==================================================

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "hubraum_l",
    "kundenzufriedenheit",
    "bundesland",
    "jahr",
    "monat_sin",
    "monat_cos",
    "wochentag_sin",
    "wochentag_cos"
]

X = df[FEATURES]
y = df["price_segment"]


# ==================================================
# 7. ONE-HOT ENCODING
# ==================================================

X = pd.get_dummies(X, drop_first=True)


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
# 9. MODEL
# ==================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


# ==================================================
# 10. TRAIN
# ==================================================

print("\nTraining price segmentation model...")

model.fit(X_train, y_train)

print("Training completed.")


# ==================================================
# 11. PREDICT
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 12. EVALUATION
# ==================================================

acc = accuracy_score(y_test, y_pred)

print("\n========== PRICE SEGMENT RESULTS ==========")
print(f"Accuracy: {acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))


# ==================================================
# 13. EXAMPLE PREDICTIONS
# ==================================================

print("\nSample predictions:")
print(pd.DataFrame({
    "true": y_test.values[:10],
    "pred": y_pred[:10]
}))