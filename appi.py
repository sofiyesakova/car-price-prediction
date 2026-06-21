from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for 
import pandas as pd
import joblib
import os
import re
import traceback
from user_prediction_system.prediction_storage import (
    save_user_prediction,
    get_all_predictions,
    get_brand_request_counts,
    get_brand_model_counts
      )
app = Flask(__name__)
app.secret_key = "autopredict-demo-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
price_model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))
satisfaction_model = joblib.load(os.path.join(BASE_DIR, "models", "customer_satisfaction_tree.pkl"))

BRANDS = ["Audi","Bmw", "Mercedes-Benz", "Opel", "Volkswagen"]

MODELS = [
    "5Er", "A4", "A6", "Astra", "C-Klasse", "Corsa",
    "E-Klasse", "E-Tron", "Eqe", "Glc", "Golf",
    "Grandland", "I4", "Id.4", "Mokka", "Passat",
    "Q5", "Tiguan", "X5"
]

FUELS = ["Benzin", "Diesel", "Elektro", "Hybrid"]
TRANSMISSIONS = ["Automatik", "Manuell"]
STATES = ["Bayern", "Berlin", "Hamburg", "Hessen", "Nrw", "Niedersachsen"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def extract_from_list(message, options, default):
    message_lower = message.lower()
    for option in options:
        if option.lower() in message_lower:
            return option
    return default


def contains_vehicle_info(message):

    message_lower = message.lower()

    has_brand = any(
        brand.lower() in message_lower
        for brand in BRANDS
    )

    has_model = any(
        model.lower() in message_lower
        for model in MODELS
    )

    has_fuel = any(
        fuel.lower() in message_lower
        for fuel in FUELS
    )

    has_transmission = any(
        transmission.lower() in message_lower
        for transmission in TRANSMISSIONS
    )

    has_engine = (
        "hubraum" in message_lower
        or "liter" in message_lower
        or re.search(r"\d+[.,]?\d*", message_lower)
    )

    return (
        has_brand
        and has_model
        and has_fuel
        and has_transmission
        and has_engine
    )

def extract_number(message, keywords, default):
    message_lower = message.lower()

    for keyword in keywords:
        pattern = rf"{keyword}\s*[:=]?\s*(\d+([.,]\d+)?)"
        match = re.search(pattern, message_lower)

        if match:
            return float(match.group(1).replace(",", "."))

    return default


def extract_vehicle_data_from_message(message):
    marke = extract_from_list(message, BRANDS, "Bmw")
    modell = extract_from_list(message, MODELS, "X5")
    kraftstoff = extract_from_list(message, FUELS, "Benzin")
    getriebe = extract_from_list(message, TRANSMISSIONS, "Automatik")
    bundesland = extract_from_list(message, STATES, "Bayern")
    wochentag = extract_from_list(message, WEEKDAYS, "Monday")

    jahr = int(extract_number(message, ["jahr", "baujahr", "year"], 2024))
    monat = int(extract_number(message, ["monat", "month"], 1))
    verkaufszahl = extract_number(message, ["verkaufszahl", "sales"], 2)
    kundenzufriedenheit = extract_number(
        message,
        ["kundenzufriedenheit", "zufriedenheit", "customer satisfaction"],
        4.0
    )

    hubraum_l = extract_number(
        message,
        ["hubraum", "hubraum_l", "engine", "motor"],
        0.0 if kraftstoff == "Elektro" else 2.0
    )

    return {
        "marke": marke,
        "modell": modell,
        "verkaufszahl": verkaufszahl,
        "kraftstoff": kraftstoff,
        "getriebe": getriebe,
        "hubraum_l": hubraum_l,
        "bundesland": bundesland,
        "kundenzufriedenheit": kundenzufriedenheit,
        "jahr": jahr,
        "monat": monat,
        "wochentag": wochentag
    }


def predict_price(car_data):
    print("DETECTED DATA:", car_data, flush=True)

    input_df = pd.DataFrame([car_data])
    print("INPUT DATAFRAME:", input_df, flush=True)

    predicted_price = price_model.predict(input_df)[0]
    return round(float(predicted_price), 2)


def predict_customer_satisfaction(car_data):
    satisfaction_features = [
        "marke",
        "modell",
        "kraftstoff",
        "getriebe",
        "hubraum_l"
    ]

    input_df = pd.DataFrame([{
        feature: car_data[feature]
        for feature in satisfaction_features
    }])

    predicted_class = int(satisfaction_model.predict(input_df)[0])

    probability = None
    if hasattr(satisfaction_model, "predict_proba"):
        probabilities = satisfaction_model.predict_proba(input_df)[0]
        classes = list(satisfaction_model.classes_)
        if 1 in classes:
            probability = round(float(probabilities[classes.index(1)]) * 100, 2)

    return {
        "class": predicted_class,
        "label": "Zufrieden" if predicted_class == 1 else "Nicht zufrieden",
        "probability_percent": probability
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/satisfaction")
def satisfaction():
    return render_template("satisfaction.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or {}
        predicted_price = predict_price(data)
        
        return jsonify({
            "status": "success",
            "predicted_price_euro": predicted_price,
            "detected_data": data
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details, flush=True)

        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500


@app.route("/api/predict-satisfaction", methods=["POST"])
def api_predict_satisfaction():
    try:
        data = request.get_json() or {}
        result = predict_customer_satisfaction(data)

        return jsonify({
            "status": "success",
            "predicted_satisfaction": result["class"],
            "predicted_satisfaction_label": result["label"],
            "satisfaction_probability_percent": result["probability_percent"],
            "detected_data": data
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details, flush=True)

        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500

@app.route("/api/chatbot", methods=["POST"])
def chatbot():

    try:
        data = request.get_json() or {}
        message = data.get("message", "")
        
        if not contains_vehicle_info(message):

          return jsonify({
        "status": "contact"  })

        car_data = extract_vehicle_data_from_message(message)

        detected_brand = car_data["marke"]
        detected_model = car_data["modell"]

        if detected_brand == "Bmw" and detected_model == "X5":
            pass

        predicted_price = predict_price(car_data)

        satisfaction_data = car_data.copy()
        satisfaction_data["preis_euro"] = predicted_price

        satisfaction_result = predict_customer_satisfaction(
            satisfaction_data
        )

        response_text = f"""
           Die KI-Analyse wurde erfolgreich erstellt.

            Geschätzter Preis:
          {predicted_price:,.0f} €

           Vorhergesagte Kundenzufriedenheit:
           {satisfaction_result['label']}
         """

        return jsonify({
            "status": "success",
            "answer": response_text
        })

    except Exception:

        return jsonify({
            "status": "contact"
        })

@app.route("/api/contact-predict", methods=["POST"])
def api_contact_predict():
    try:
        data = request.get_json() or {}

        name = data.get("name", "")
        email = data.get("email", "")
        subject = data.get("subject", "")
        message = data.get("message", "")

        if not contains_vehicle_info(message):
         return jsonify({
        "status": "error",
        "message": """
        Die KI konnte keine zuverlässige Preis- und Zufriedenheitsprognose erstellen.

        Bitte beschreiben Sie Ihr Fahrzeug mit den folgenden Informationen:

        • Marke
        • Modell
        • Kraftstoff
        • Getriebe
        • Hubraum

        Beispiel:

        Ich möchte eine Prognose für einen Audi A4 mit Benzin,
        Automatikgetriebe und 2.0 Liter Hubraum erhalten.
        """,
        "details": ""
          }), 400

        car_data = extract_vehicle_data_from_message(message)
        predicted_price = predict_price(car_data)

        satisfaction_data = car_data.copy()
        satisfaction_data["preis_euro"] = predicted_price

        satisfaction_result = predict_customer_satisfaction(satisfaction_data)

        save_user_prediction(
            name=name,
            email=email,
            subject=subject,
            message=message,
            car_data=car_data,
            predicted_price_euro=predicted_price,
            predicted_satisfaction_label=satisfaction_result["label"],
            satisfaction_probability_percent=satisfaction_result["probability_percent"]
        )

        return jsonify({
            "status": "success",
            "answer": "Vielen Dank für Ihre Anfrage. Die KI-Analyse wurde erfolgreich erstellt.",
            "predicted_price_euro": predicted_price,
            "predicted_satisfaction_label": satisfaction_result["label"],
            "satisfaction_probability_percent": satisfaction_result["probability_percent"],
            "detected_data": car_data
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details, flush=True)

        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500


@app.route("/prediction-history")
def prediction_history():

    predictions = get_all_predictions()

    clean_predictions = []

    for row in predictions:

        clean_predictions.append({
            "created_at": row["created_at"],
            "name": row["name"],
            "email": row["email"],
            "subject": row["subject"],
            "message": row["message"],

            "marke": row["marke"],
            "modell": row["modell"],

            "predicted_price_euro":
                row["predicted_price_euro"],

            "predicted_satisfaction":
                row["predicted_satisfaction_label"],

            "probability":
                row["satisfaction_probability_percent"]
        })

    return jsonify(clean_predictions)

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    error = ""

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))

        error = "Falscher Benutzername oder falsches Passwort."

    return render_template("admin_login.html", error=error)


@app.route("/admin")
def admin():

    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    predictions = get_all_predictions()

    brand_counts = get_brand_request_counts()

    brand_model_counts = get_brand_model_counts()

    return render_template(
        "admin.html",

        predictions=predictions,

        brand_counts=brand_counts,

        brand_model_counts=brand_model_counts
    )

@app.route("/admin-logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

@app.route("/admin/export-csv")
def export_predictions_csv():
    predictions = get_all_predictions()

    def generate():
        header = [
            "id", "created_at", "name", "email", "subject", "message",
            "marke", "modell", "predicted_price_euro",
            "predicted_satisfaction_label",
            "satisfaction_probability_percent"
        ]

        yield ",".join(header) + "\n"

        for row in predictions:
            values = [
                row.get("id", ""),
                row.get("created_at", ""),
                row.get("name", ""),
                row.get("email", ""),
                row.get("subject", ""),
                row.get("message", ""),
                row.get("marke", ""),
                row.get("modell", ""),
                row.get("predicted_price_euro", ""),
                row.get("predicted_satisfaction_label", ""),
                row.get("satisfaction_probability_percent", "")
            ]

            yield ",".join([f'"{str(v).replace(chr(34), chr(34)+chr(34))}"' for v in values]) + "\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=user_predictions.csv"
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
