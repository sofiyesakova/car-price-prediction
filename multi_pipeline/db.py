from sqlalchemy import create_engine, text
from .config import *

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def save_to_db(
    data,
    price,
    satisfaction,
    category_2,
    category_3,
    datum
):

    query = text("""
        INSERT INTO email_predictions(
            datum,

            marke,
            modell,
            kraftstoff,
            getriebe,
            bundesland,

            verkaufszahl,
            hubraum_l,

            predicted_price,
            predicted_zufriedenheit,
            predicted_category_2,
            predicted_category_3
        )
        VALUES(
            :datum,

            :marke,
            :modell,
            :kraftstoff,
            :getriebe,
            :bundesland,

            :verkaufszahl,
            :hubraum_l,

            :price,
            :satisfaction,
            :category_2,
            :category_3
        )
    """)

    payload = {
        "datum": datum,

        "marke": data.get("marke") or "unknown",
        "modell": data.get("modell") or "unknown",
        "kraftstoff": data.get("kraftstoff") or "unknown",
        "getriebe": data.get("getriebe") or "unknown",
        "bundesland": data.get("bundesland") or "unknown",

        "verkaufszahl": float(data.get("verkaufszahl") or 0),
        "hubraum_l": float(data.get("hubraum_l") or 0),

        "price": float(price),
        "satisfaction": int(satisfaction),
        "category_2": int(category_2),
        "category_3": int(category_3)
    }

    with engine.begin() as conn:
        conn.execute(query, payload)