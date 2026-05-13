from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__, static_folder=".")

MODEL_PATH = "flight_price_prediction_model.pkl"
model = joblib.load(MODEL_PATH)

FARE_MIN = 1307
FARE_MAX = 143019
DURATION_MIN = 0.75
DURATION_MAX = 43.5833
DAYS_LEFT_MIN = 1
DAYS_LEFT_MAX = 50

AIRLINES = [
    "Air India",
    "AirAsia",
    "AkasaAir",
    "AllianceAir",
    "GO FIRST",
    "Indigo",
    "SpiceJet",
    "StarAir",
    "Vistara",
]
CITIES = ["Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
TIME_WINDOWS = ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

FEATURE_COLUMNS = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "Class",
            "Total_stops",
            "Duration_in_hours",
            "Days_left",
            "Journey_day_Friday",
            "Journey_day_Monday",
            "Journey_day_Saturday",
            "Journey_day_Sunday",
            "Journey_day_Thursday",
            "Journey_day_Tuesday",
            "Journey_day_Wednesday",
            "Airline_Air India",
            "Airline_AirAsia",
            "Airline_AkasaAir",
            "Airline_AllianceAir",
            "Airline_GO FIRST",
            "Airline_Indigo",
            "Airline_SpiceJet",
            "Airline_StarAir",
            "Airline_Vistara",
            "Source_Ahmedabad",
            "Source_Bangalore",
            "Source_Chennai",
            "Source_Delhi",
            "Source_Hyderabad",
            "Source_Kolkata",
            "Source_Mumbai",
            "Destination_Ahmedabad",
            "Destination_Bangalore",
            "Destination_Chennai",
            "Destination_Delhi",
            "Destination_Hyderabad",
            "Destination_Kolkata",
            "Destination_Mumbai",
            "Departure_12 PM - 6 PM",
            "Departure_6 AM - 12 PM",
            "Departure_After 6 PM",
            "Departure_Before 6 AM",
            "Arrival_12 PM - 6 PM",
            "Arrival_6 AM - 12 PM",
            "Arrival_After 6 PM",
            "Arrival_Before 6 AM",
            "journey_day",
            "journey_month",
            "is_weekend",
        ],
    )
)


def scale(value, low, high):
    return (value - low) / (high - low)


def one_hot(prefix, options, selected):
    return {f"{prefix}_{option}": 1 if option == selected else 0 for option in options}


def choice_from_index(values, form_key):
    raw_value = int(request.form[form_key])
    try:
        return values[raw_value - 1]
    except IndexError as exc:
        raise ValueError(f"Invalid value for {form_key}") from exc


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/styles.css")
def stylesheet():
    return send_from_directory(".", "styles.css", mimetype="text/css")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        airline = choice_from_index(AIRLINES, "airline")
        source = choice_from_index(CITIES, "source")
        destination = choice_from_index(CITIES, "destination")
        departure = choice_from_index(TIME_WINDOWS, "departure")
        arrival = choice_from_index(TIME_WINDOWS, "arrival")

        flight_class = int(request.form["flight_class"])
        total_stops = int(request.form["total_stops"])
        duration_hours = float(request.form["duration_hours"])
        days_left = int(request.form["days_left"])
        journey_date = datetime.strptime(request.form["journey_date"], "%Y-%m-%d")
        weekday_name = WEEKDAYS[journey_date.weekday()]

        if source == destination:
            raise ValueError("Source and destination must be different cities.")
        if not DURATION_MIN <= duration_hours <= DURATION_MAX:
            raise ValueError("Duration must be between 0.75 and 43.58 hours.")
        if not DAYS_LEFT_MIN <= days_left <= DAYS_LEFT_MAX:
            raise ValueError("Days left must be between 1 and 50.")

        input_data = {
            "Class": flight_class,
            "Total_stops": total_stops,
            "Duration_in_hours": scale(duration_hours, DURATION_MIN, DURATION_MAX),
            "Days_left": scale(days_left, DAYS_LEFT_MIN, DAYS_LEFT_MAX),
            "journey_day": journey_date.day,
            "journey_month": journey_date.month,
            "is_weekend": 1 if journey_date.weekday() >= 5 else 0,
            **one_hot("Journey_day", WEEKDAYS, weekday_name),
            **one_hot("Airline", AIRLINES, airline),
            **one_hot("Source", CITIES, source),
            **one_hot("Destination", CITIES, destination),
            **one_hot("Departure", TIME_WINDOWS, departure),
            **one_hot("Arrival", TIME_WINDOWS, arrival),
        }

        input_df = pd.DataFrame([input_data]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
        scaled_prediction = float(model.predict(input_df)[0])
        actual_fare = (scaled_prediction * (FARE_MAX - FARE_MIN)) + FARE_MIN
        actual_fare = max(FARE_MIN, min(actual_fare, FARE_MAX))

        return jsonify(
            {
                "prediction": round(actual_fare, 2),
                "formatted_prediction": f"₹{actual_fare:,.2f}",
                "route": f"{source} to {destination}",
                "airline": airline,
                "journey_day": weekday_name,
                "confidence_note": "Estimate from the saved XGBoost model",
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
