import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/random_forest.pkl")

preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

# =========================
# GET ENCODED FEATURE NAMES
# =========================

encoded_features = preprocessor.get_feature_names_out()
importances = regressor.feature_importances_

importance_df = pd.DataFrame({
    "Encoded Feature": encoded_features,
    "Importance": importances
})

# =========================
# GROUP FEATURES BY ORIGINAL NAME
# =========================

def get_original_feature_name(encoded_name):
    # Example: cat__marke_Bmw -> marke
    name = encoded_name.replace("cat__", "")
    return name.split("_")[0]

importance_df["Original Feature"] = importance_df["Encoded Feature"].apply(
    get_original_feature_name
)

grouped_importance = (
    importance_df
    .groupby("Original Feature")["Importance"]
    .sum()
    .reset_index()
)

grouped_importance["Importance (%)"] = (
    grouped_importance["Importance"] * 100
).round(2)

grouped_importance = grouped_importance.sort_values(
    by="Importance (%)",
    ascending=False
)

# =========================
# OUTPUT
# =========================

print("\n===== GROUPED FEATURE IMPORTANCE =====")
print(grouped_importance[["Original Feature", "Importance (%)"]])

grouped_importance.to_csv(
    "model_evaluation/feature_importance_grouped.csv",
    index=False
)

print("\nSaved: model_evaluation/feature_importance_grouped.csv")