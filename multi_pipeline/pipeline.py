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


def run_pipeline():

    try:
        print("Starting pipeline...\n")

        # ==================================================
        # EMAIL
        # ==================================================

        email = read_latest_email()

        if not email:
            print("No new emails")
            return

        print("Email received ✔")

        body = email["body"]
        received_date = email["received_datetime"]

        # ==================================================
        # 1. EXTRACT
        # ==================================================

        data = extract_with_llm(body)
        print("Data extracted ✔")
        print("RAW DATA:", data)

        # ==================================================
        # 2. TIME FEATURES
        # ==================================================

        data["jahr"] = received_date.year
        data["monat"] = received_date.month
        data["wochentag"] = received_date.weekday()

        # ==================================================
        # 3. CLEANING
        # ==================================================

        # categorical safety
        for k in ["marke", "modell", "kraftstoff", "getriebe", "bundesland"]:
            if not data.get(k):
                data[k] = "unknown"

        # numeric safety
        for k in ["verkaufszahl", "hubraum_l"]:
            try:
                data[k] = float(data.get(k) or 0)
            except:
                data[k] = 0

        # ==================================================
        # 4. ML
        # ==================================================

        price = predict_price(data)

        sat_pred = predict_satisfaction(data)
        sat_text = satisfaction_text(sat_pred)

        cat2 = predict_price_category_2(data)
        cat2_text = price_category_text_2(cat2)

        cat3 = predict_price_category_3(data)
        cat3_text = price_category_text_3(cat3)

        # ==================================================
        # 5. OUTPUT
        # ==================================================

        print("\n===== RESULTS =====")
        print(f"Price: {price:.2f} €")
        print(f"Kundenfreundlichkeit: {sat_text}")
        print(f"Price Category (2-class): {cat2_text}")
        print(f"Price Category (3-class): {cat3_text}")

        # ==================================================
        # 6. DB
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

    except Exception as e:
        print("\n❌ PIPELINE ERROR")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        raise


if __name__ == "__main__":
    run_pipeline()