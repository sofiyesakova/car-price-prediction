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

# RANDOM FOREST (PRICE)
price_model = joblib.load(MODEL_DIR / "random_forest_5features.pkl")

# CLASSIFICATION (SATISFACTION)
satisfaction_model = joblib.load(
    MODEL_DIR / "customer_satisfaction_tree.pkl"
)

# NEW: PRICE CATEGORY MODELS
price_category_model_2 = joblib.load(
    MODEL_DIR / "random_forest_fahrzeugpreiskategorie_2klassen.pkl"
)

price_category_model_3 = joblib.load(
    MODEL_DIR / "random_forest_fahrzeugpreiskategorie_3klassen.pkl"
)

# ==================================================
# FEATURE SCHEMA (ALL ML MODELS USE SAME INPUT)
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

# ==================================================
# CLEAN DATA
# ==================================================

def clean_data(data: dict) -> dict:
    return {
        "marke": str(data.get("marke", "unknown")),
        "modell": str(data.get("modell", "unknown")),
        "kraftstoff": str(data.get("kraftstoff", "unknown")),
        "getriebe": str(data.get("getriebe", "unknown")),
        "bundesland": str(data.get("bundesland", "unknown")),
        "wochentag": str(data.get("wochentag", "unknown")),

        "verkaufszahl": float(data.get("verkaufszahl") or 0),
        "hubraum_l": float(data.get("hubraum_l") or 0),
        "jahr": int(data.get("jahr") or 0),
        "monat": int(data.get("monat") or 0),
    }

# ==================================================
# DATAFRAME
# ==================================================

def build_df(data: dict) -> pd.DataFrame:
    clean = clean_data(data)
    return pd.DataFrame([{f: clean.get(f, None) for f in FEATURES}])

# ==================================================
# PRICE MODEL
# ==================================================

def predict_price(data: dict) -> float:
    df = build_df(data)
    return float(price_model.predict(df)[0])

# ==================================================
# SATISFACTION MODEL
# ==================================================

def predict_satisfaction(data: dict) -> int:
    df = build_df(data)
    return int(satisfaction_model.predict(df)[0])

def satisfaction_text(pred: int) -> str:
    return (
        "Der Kunde wird wahrscheinlich zufrieden sein."
        if pred == 1
        else "Der Kunde wird wahrscheinlich nicht zufrieden sein."
    )

# ==================================================
# PRICE CATEGORY - 2 CLASSES
# ==================================================

def predict_price_category_2(data: dict) -> int:
    df = build_df(data)
    return int(price_category_model_2.predict(df)[0])

def price_category_text_2(pred: int) -> str:
    return (
        "Hochpreis-Kategorie"
        if pred == 1
        else "Nicht Hochpreis-Kategorie"
    )

# ==================================================
# PRICE CATEGORY - 3 CLASSES
# ==================================================

def predict_price_category_3(data: dict) -> int:
    df = build_df(data)
    return int(price_category_model_3.predict(df)[0])

def price_category_text_3(pred: int) -> str:

    mapping = {
        0: "Niedrigpreis",
        1: "Mittelklasse",
        2: "Hochpreis"
    }

    return mapping.get(pred, "Unbekannt")