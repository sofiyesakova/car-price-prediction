import joblib
from pathlib import Path
import pandas as pd

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

# ==================================================
# LOAD MODELS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

price_model = joblib.load(MODEL_DIR / "linear_regression.pkl")
satisfaction_model = joblib.load(MODEL_DIR / "customer_satisfaction_tree.pkl")

# ==================================================
# FEATURE SCHEMA
# ==================================================

FEATURES = [
    "marke",
    "modell",
    "kraftstoff",
    "getriebe",
    "hubraum_l",
    "verkaufszahl"
]

# ==================================================
# CLEAN DATA
# ==================================================

def clean_data(data: dict) -> dict:
    return {
        "marke": str(data.get("marke", "unknown")),
        "modell": str(data.get("modell", "unknown")),
        "kraftstoff": str(data.get("kraftstoff", "unknown")),
        "getriebe": str(data.get("getriebe", "unknown")),
        "hubraum_l": float(data.get("hubraum_l") or 0),
        "verkaufszahl": float(data.get("verkaufszahl") or 0),
    }

# ==================================================
# DATAFRAME
# ==================================================

def build_df(data: dict) -> pd.DataFrame:
    clean = clean_data(data)
    return pd.DataFrame([{f: clean.get(f, None) for f in FEATURES}])

# ==================================================
# PRICE
# ==================================================

def predict_price(data: dict) -> float:
    df = build_df(data)
    return float(price_model.predict(df)[0])

# ==================================================
# SATISFACTION (ML OUTPUT = INT ONLY)
# ==================================================

def predict_satisfaction(data: dict) -> int:
    df = build_df(data)
    return int(satisfaction_model.predict(df)[0])

# ==================================================
# HUMAN TEXT (UI ONLY)
# ==================================================

def satisfaction_text(pred: int) -> str:
    if pred == 1:
        return "Der Kunde wird wahrscheinlich zufrieden sein."
    return "Der Kunde wird wahrscheinlich nicht zufrieden sein."