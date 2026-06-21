import sqlite3
import os
from datetime import datetime

import psycopg2
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "user_predictions.db")


def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,

            name TEXT,
            email TEXT,
            subject TEXT,
            message TEXT,

            marke TEXT,
            modell TEXT,
            kraftstoff TEXT,
            getriebe TEXT,
            bundesland TEXT,
            wochentag TEXT,

            verkaufszahl REAL,
            hubraum_l REAL,
            kundenzufriedenheit REAL,
            jahr INTEGER,
            monat INTEGER,

            predicted_price_euro REAL,
            predicted_satisfaction_label TEXT,
            satisfaction_probability_percent REAL
        )
    """)

    conn.commit()
    conn.close()


def save_to_postgres(
    name,
    email,
    subject,
    message,
    car_data,
    predicted_price_euro,
    predicted_satisfaction_label,
    satisfaction_probability_percent
):
    print("TRYING TO CONNECT TO POSTGRES...", flush=True)

    conn = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME")
    )

    print("POSTGRES CONNECTED", flush=True)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_prediction_history (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP,

            name TEXT,
            email TEXT,
            subject TEXT,
            message TEXT,

            marke TEXT,
            modell TEXT,
            kraftstoff TEXT,
            getriebe TEXT,
            hubraum_l REAL,

            predicted_price_euro REAL,
            predicted_satisfaction_label TEXT,
            satisfaction_probability_percent REAL
        )
    """)

    cursor.execute("""
        INSERT INTO user_prediction_history (
            created_at,
            name,
            email,
            subject,
            message,
            marke,
            modell,
            kraftstoff,
            getriebe,
            hubraum_l,
            predicted_price_euro,
            predicted_satisfaction_label,
            satisfaction_probability_percent
        )
        VALUES (
            NOW(), %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s
        )
    """, (
        name,
        email,
        subject,
        message,
        car_data.get("marke"),
        car_data.get("modell"),
        car_data.get("kraftstoff"),
        car_data.get("getriebe"),
        car_data.get("hubraum_l"),
        predicted_price_euro,
        predicted_satisfaction_label,
        satisfaction_probability_percent
    ))

    conn.commit()
    cursor.close()
    conn.close()


def save_user_prediction(
    name,
    email,
    subject,
    message,
    car_data,
    predicted_price_euro,
    predicted_satisfaction_label,
    satisfaction_probability_percent
):
    create_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("POSTGRES DATA:", {
    "marke": car_data.get("marke"),
    "modell": car_data.get("modell"),
    "kraftstoff": car_data.get("kraftstoff"),
    "getriebe": car_data.get("getriebe"),
    "hubraum_l": car_data.get("hubraum_l")
}, flush=True)
    cursor.execute("""
        INSERT INTO user_predictions (
            created_at,
            name,
            email,
            subject,
            message,
            marke,
            modell,
            kraftstoff,
            getriebe,
            bundesland,
            wochentag,
            verkaufszahl,
            hubraum_l,
            kundenzufriedenheit,
            jahr,
            monat,
            predicted_price_euro,
            predicted_satisfaction_label,
            satisfaction_probability_percent
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name,
        email,
        subject,
        message,
        car_data.get("marke"),
        car_data.get("modell"),
        car_data.get("kraftstoff"),
        car_data.get("getriebe"),
        car_data.get("bundesland"),
        car_data.get("wochentag"),
        car_data.get("verkaufszahl"),
        car_data.get("hubraum_l"),
        car_data.get("kundenzufriedenheit"),
        car_data.get("jahr"),
        car_data.get("monat"),
        predicted_price_euro,
        predicted_satisfaction_label,
        satisfaction_probability_percent
    ))

    conn.commit()
    conn.close()

    print("SQLITE SAVE SUCCESSFUL", flush=True)

    try:
        save_to_postgres(
            name=name,
            email=email,
            subject=subject,
            message=message,
            car_data=car_data,
            predicted_price_euro=predicted_price_euro,
            predicted_satisfaction_label=predicted_satisfaction_label,
            satisfaction_probability_percent=satisfaction_probability_percent
        )

        print("POSTGRES SAVE SUCCESSFUL", flush=True)

    except Exception as e:
        print("POSTGRES SAVE FAILED:", e, flush=True)


def get_all_predictions():
    create_table()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM user_predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_brand_request_counts():
    create_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT marke, COUNT(*) as request_count
        FROM user_predictions
        WHERE marke IS NOT NULL
        GROUP BY marke
        ORDER BY request_count DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "marke": row[0],
            "request_count": row[1]
        }
        for row in rows
    ]


def get_brand_model_counts():
    create_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT marke, modell, COUNT(*) as request_count
        FROM user_predictions
        WHERE marke IS NOT NULL
          AND modell IS NOT NULL
        GROUP BY marke, modell
        ORDER BY marke, request_count DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "marke": row[0],
            "modell": row[1],
            "request_count": row[2]
        }
        for row in rows
    ]