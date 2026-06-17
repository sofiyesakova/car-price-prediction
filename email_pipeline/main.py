from email_pipeline.email_reader import read_latest_email
from email_pipeline.llm_extractor import extract_with_llm
from email_pipeline.predictor import predict
from email_pipeline.db_writer import save_prediction


def main():

    print("Reading email...")

    email_text = read_latest_email()

    if not email_text:
        print("No unread emails.")
        return

    print("\nEMAIL:")
    print(email_text)

    print("\nExtracting...")

    data = extract_with_llm(
        email_text
    )

    print("\nJSON:")
    print(data)

    print("\nPredicting...")

    prediction = predict(data)

    print(
        f"\nPredicted price: {prediction:.2f} €"
    )

    save_prediction(
        data,
        prediction
    )

    print(
        "\nSaved to PostgreSQL."
    )


if __name__ == "__main__":
    main()