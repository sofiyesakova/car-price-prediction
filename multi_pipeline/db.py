from sqlalchemy import create_engine, text
from .config import *

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def save_to_db(data, price, satisfaction):

    query = text("""
        INSERT INTO email_predictions(
            datum,
            marke,
            modell,
            kraftstoff,
            getriebe,
            hubraum_l,
            verkaufszahl,
            kundenzufriedenheit,
            predicted_price,
            predicted_zufriedenheit
        )
        VALUES(
            :datum,
            :marke,
            :modell,
            :kraftstoff,
            :getriebe,
            :hubraum_l,
            :verkaufszahl,
            :kundenzufriedenheit,
            :price,
            :satisfaction
        )
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            **data,
            "price": price,
            "satisfaction": satisfaction
        })