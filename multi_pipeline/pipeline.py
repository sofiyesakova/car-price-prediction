from .email_reader import read_latest_email
from .llm_extractor import extract_with_llm
from .ml_models import (
    predict_price,
    predict_satisfaction,
    satisfaction_text
)
from .db import save_to_db


def run_pipeline():

    print("Starting pipeline...\n")

    email = read_latest_email()

    if not email:
        print("No new emails")
        return

    print("Email received ✔")

    data = extract_with_llm(email)
    print("Data extracted ✔")

    price = predict_price(data)

    sat_pred = predict_satisfaction(data)
    sat_text = satisfaction_text(sat_pred)

    print("\n===== RESULTS =====")
    print(f"Price: {price:.2f} €")
    print(f"Satisfaction: {sat_text}")

    # IMPORTANT: save integer, NOT text
    save_to_db(data, price, sat_pred)

    print("\nSaved to DB ✔")


if __name__ == "__main__":
    run_pipeline()