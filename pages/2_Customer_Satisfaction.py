import streamlit as st
import pandas as pd
import joblib

# ==================================================

# LOAD MODEL

# ==================================================

model = joblib.load(
"models/customer_satisfaction_tree.pkl"
)

# ==================================================

# PAGE CONFIG

# ==================================================

st.set_page_config(
page_title="Kundenzufriedenheit Vorhersage",
page_icon="😊"
)

st.title("😊 Kundenzufriedenheit Vorhersage")

st.write(
"""
Geben Sie Fahrzeuginformationen ein und prediktieren Sie,
ob der Kunde wahrscheinlich zufrieden sein wird.
"""
)

# ==================================================

# INPUT FIELDS

# ==================================================

marke = st.selectbox("Marke", ["Bmw", "Mercedes-Benz", "Volkswagen", "Opel"])
modell = st.text_input("Modell", "C-Klasse")

kraftstoff = st.selectbox(
"Kraftstoff",
[
"Benzin",
"Diesel",
"Hybrid",
"Elektro"
]
)

getriebe = st.selectbox(
"Getriebe",
[
"Manuell",
"Automatik"
]
)

hubraum_l = st.number_input(
"Hubraum (L)",
min_value=0.0,
max_value=8.0,
value=2.0,
step=0.1
)

# ==================================================

# PREDICTION

# ==================================================

if st.button("Kundenfreude vorhersagen"):

    input_data = pd.DataFrame(
        {
            "marke": [marke],
            "modell": [modell],
            "kraftstoff": [kraftstoff],
            "getriebe": [getriebe],
            "hubraum_l": [hubraum_l]
        }
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:

        st.success(
            "✅ Kunde wird wahrscheinlich zufrieden sein."
        )

    else:

        st.error(
            "❌ Kunde wird wahrscheinlich NICHT zufrieden sein."
        )