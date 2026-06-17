from .email_reader import get_mock_email
from .llm_extractor import extract_with_llm
from .ml_model import predict
from .db import save_to_db


import json

def parse_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        raise ValueError("Invalid JSON from LLM")


def run_pipeline():

    email = get_mock_email()

    print("EMAIL:")
    print(email)

    print("\nLLM extracting...")

    llm_output = extract_with_llm(email)

    print("\nLLM OUTPUT:")
    print(llm_output)

    data = extract_json(llm_output)

    print("\nPARSED JSON:")
    print(data)

    prediction = predict(data)

    print("\nPREDICTION:", prediction)

    save_to_db(data, prediction)

    print("\nSaved to DB ✔")


if __name__ == "__main__":
    run_pipeline()