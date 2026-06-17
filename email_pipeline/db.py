from sqlalchemy import create_engine, text
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def save_to_db(data, prediction):

    query = text("""
        INSERT INTO predictions (
            datum, marke, modell, kraftstoff,
            getriebe, hubraum_l, verkaufszahl,
            kundenzufriedenheit, prediction
        )
        VALUES (
            :datum, :marke, :modell, :kraftstoff,
            :getriebe, :hubraum_l, :verkaufszahl,
            :kundenzufriedenheit, :prediction
        )
    """)

    params = {
        **data,
        "prediction": float(prediction)
    }

    with engine.begin() as conn:
        conn.execute(query, params)