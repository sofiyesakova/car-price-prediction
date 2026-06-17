from sqlalchemy import create_engine, text
from .config import *


engine = create_engine(
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def save_prediction(data, prediction):

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
            predicted_price
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
            :prediction
        )
    """)

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                **data,
                "prediction": prediction
            }
        )