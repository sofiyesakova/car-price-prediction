import joblib
import pandas as pd


model = joblib.load("models/linear_regression.pkl")


def predict(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return float(prediction[0])