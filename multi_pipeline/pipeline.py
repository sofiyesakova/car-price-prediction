from .email_reader import read_latest_email
from .llm_extractor import extract_with_llm
from .ml_models import (
    predict_price,
    predict_satisfaction,
    satisfaction_text,
    predict_price_category_2,
    price_category_text_2,
    predict_price_category_3,
    price_category_text_3
)
from .db import save_to_db
from datetime import datetime


def run_pipeline():

    print("Starting pipeline...\n")

    email = read_latest_email()

    if not email:
        print("No new emails")
        return

    print("Email received ✔")

    # ==================================================
    # 1. EXTRACT
    # ==================================================

    body = email["body"]
    received_date = email["received_datetime"]

    data = extract_with_llm(body)
    print("Data extracted ✔")

    # ==================================================
    # 2. TIME FEATURES
    # ==================================================

    data["jahr"] = received_date.year
    data["monat"] = received_date.month
    data["wochentag"] = received_date.strftime("%A")

    # ==================================================
    # 3. ML
    # ==================================================

    price = predict_price(data)

    sat_pred = predict_satisfaction(data)
    sat_text = satisfaction_text(sat_pred)

    cat2 = predict_price_category_2(data)
    cat2_text = price_category_text_2(cat2)

    cat3 = predict_price_category_3(data)
    cat3_text = price_category_text_3(cat3)

    # ==================================================
    # 4. OUTPUT
    # ==================================================

    print("\n===== RESULTS =====")
    print(f"Price: {price:.2f} €")
    print(f"Kundenfreundlichkeit: {sat_text}")
    print(f"Price Category (2-class): {cat2_text}")
    print(f"Price Category (3-class): {cat3_text}")

    # ==================================================
    # 5. DB
    # ==================================================

    save_to_db(
        data,
        price,
        sat_pred,
        cat2,
        cat3,
        received_date
    )

    print("\nSaved to DB ✔")


if __name__ == "__main__":
    run_pipeline()