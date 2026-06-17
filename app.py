import streamlit as st
import pandas as pd
import joblib


# ==================================================
# 1. LOAD MODELS
# ==================================================

linear_model = joblib.load("models/linear_regression.pkl")
rf_model = joblib.load("models/random_forest.pkl")
gb_model = joblib.load("models/gradient_boosting.pkl")

models = {
    "Linear Regression": linear_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}


# ==================================================
# 2. UI
# ==================================================

st.title("🚗 Car Price Prediction")
st.write("Choose model and enter car parameters")


# ==================================================
# 3. MODEL SELECTION
# ==================================================

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[model_name]


# ==================================================
# 4. INPUT FIELDS
# ==================================================

marke = st.selectbox("Marke", ["BMW", "Mercedes-Benz", "Volkswagen", "Opel"])
modell = st.text_input("Modell", "C-Klasse")
# kraftstoff = st.selectbox("Kraftstoff", ["Benzin", "Diesel", "Elektro", "Hybrid"])
# getriebe = st.selectbox("Getriebe", ["Automatik", "Manuell"])
# hubraum_l = st.number_input("Hubraum (L)", min_value=0.0, max_value=10.0, value=2.0)


# ==================================================
# 5. PREDICTION
# ==================================================

if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "marke": marke,
        "modell": modell,
        # "kraftstoff": kraftstoff,
        # "getriebe": getriebe,
        # "hubraum_l": hubraum_l
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted price ({model_name}): € {prediction:,.2f}")