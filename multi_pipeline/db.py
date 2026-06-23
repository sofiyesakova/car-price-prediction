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

            jahr,
            monat,
            wochentag,

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

            :jahr,
            :monat,
            :wochentag,

            :price,
            :satisfaction,
            :category_2,
            :category_3
        )
    """)

    payload = {
        "datum": datum,

        "marke": data.get("marke"),
        "modell": data.get("modell"),
        "kraftstoff": data.get("kraftstoff"),
        "getriebe": data.get("getriebe"),
        "bundesland": data.get("bundesland"),

        "verkaufszahl": data.get("verkaufszahl"),
        "hubraum_l": data.get("hubraum_l"),

        "jahr": data.get("jahr"),
        "monat": data.get("monat"),
        "wochentag": data.get("wochentag"),

        "price": price,
        "satisfaction": satisfaction,
        "category_2": category_2,
        "category_3": category_3
    }

    with engine.begin() as conn:
        conn.execute(query, payload)