import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)

import joblib

from db.data_loader import load_data

# ==================================================

# 1. LOAD DATA

# ==================================================

df = load_data()

print(f"Data shape: {df.shape}")

# ==================================================

# 2. CREATE TARGET

# ==================================================

# 1 = zufrieden

# 0 = nicht zufrieden

df["zufrieden"] = (
df["kundenzufriedenheit"] >= 3.5
).astype(int)

TARGET = "zufrieden"

# ==================================================

# 3. FEATURE SETS

# ==================================================

FEATURE_SETS = {
"basic": [
    "marke",
    "modell"
],

"extended": [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "hubraum_l"
],

"all": [
    "marke",
    "modell",
    "verkaufszahl",
    "kraftstoff",
    "getriebe",
    "hubraum_l",
    "bundesland",
    "jahr",
    "monat",
    "wochentag"
]
}

SELECTED_FEATURE_SET = "extended"

selected_features = FEATURE_SETS[SELECTED_FEATURE_SET]

X = df[selected_features]
y = df[TARGET]

print(f"\nUsing feature set: {SELECTED_FEATURE_SET}")
print(selected_features)

# ==================================================

# 4. DEFINE FEATURE TYPES

# ==================================================

ALL_NUMERIC_FEATURES = [
"verkaufszahl",
"hubraum_l",
"jahr",
"monat"
]

ALL_CATEGORICAL_FEATURES = [
"marke",
"modell",
"kraftstoff",
"getriebe",
"bundesland",
"wochentag"
]

numeric_features = [
feature
for feature in selected_features
if feature in ALL_NUMERIC_FEATURES
]

categorical_features = [
feature
for feature in selected_features
if feature in ALL_CATEGORICAL_FEATURES
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ==================================================

# 5. PREPROCESSING

# ==================================================

transformers = []

if numeric_features:
    transformers.append(
        (
            "num",
            StandardScaler(),
            numeric_features
        )
)

if categorical_features:
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

# ==================================================

# 5. PREPROCESSING

# ==================================================

transformers = []

if numeric_features:
    transformers.append(
    (
        "num",
        StandardScaler(),
        numeric_features
    )
)

if categorical_features:
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
transformers=transformers
)



model = Pipeline(
steps=[
("preprocessor", preprocessor),
(
"classifier",
DecisionTreeClassifier(
max_depth=5,
random_state=42
)
)
]
)



X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# ==================================================

# 8. TRAIN MODEL

# ==================================================

model.fit(X_train, y_train)

print("\nDecision Tree training completed.")

# ==================================================

# 9. PREDICTIONS

# ==================================================

y_pred = model.predict(X_test)

# ==================================================

# 10. EVALUATION

# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
y_test,
y_pred
)

recall = recall_score(
y_test,
y_pred
)

f1 = f1_score(
y_test,
y_pred
)

print("\n===== DECISION TREE RESULTS =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================================================

# 11. SAVE MODEL

# ==================================================

joblib.dump(
model,
"models/customer_satisfaction_tree.pkl"
)

print("\nModel saved.")
