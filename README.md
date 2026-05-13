# Airline Fare Prediction

A machine learning web application that predicts Indian domestic flight fares from route, airline, cabin class, stops, departure and arrival windows, journey date, travel duration, and booking lead time.

The project includes a trained XGBoost regression model, a Flask prediction API, and a responsive booking-console style frontend.

## Features

- Predicts flight fares in INR using a saved XGBoost model.
- Browser-based itinerary form with live route summary.
- Flask `/predict` endpoint for model inference.
- Input validation for city routes, duration, and days left.
- Includes cleaned and scraped airline fare datasets.
- Jupyter notebook with preprocessing, model training, comparison, and final model export.

## Tech Stack

- Python
- Flask
- Pandas
- Joblib
- XGBoost
- HTML, CSS, and vanilla JavaScript

## Project Structure

```text
.
|-- airline.ipynb                         # Data analysis, preprocessing, model training, and model export
|-- Cleaned_dataset.csv                   # Cleaned training dataset
|-- Scraped_dataset.csv                   # Original scraped dataset
|-- flight_price_prediction_model.pkl     # Saved XGBoost model used by the Flask app
|-- index.html                            # Frontend booking console
|-- styles.css                            # Frontend styling
|-- server.py                             # Flask server and prediction endpoint
|-- requirements.txt                      # Python dependencies
`-- archive.zip                           # Archived dataset/source files
```

## Dataset

The cleaned dataset contains flight fare records with columns such as:

- `Date_of_journey`
- `Journey_day`
- `Airline`
- `Flight_code`
- `Class`
- `Source`
- `Departure`
- `Total_stops`
- `Arrival`
- `Destination`
- `Duration_in_hours`
- `Days_left`
- `Fare`

The app currently supports these cities:

- Ahmedabad
- Bangalore
- Chennai
- Delhi
- Hyderabad
- Kolkata
- Mumbai

## Model Summary

The notebook compares multiple regression models, including Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP, and Extra Trees.

The final deployed model is an XGBoost Regressor saved as:

```text
flight_price_prediction_model.pkl
```

According to the notebook project report, the final XGBoost model achieved approximately `95.58%` R2 score on the test set.

## Setup

### 1. Clone or open the project

```bash
cd airline_prediciotn
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

On Linux or macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

Start the Flask server:

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:5000
```

Fill in the itinerary form and click **Search Fare** to get a predicted fare.

## API Usage

The app exposes one prediction endpoint:

```http
POST /predict
```

Expected form fields:

| Field | Description |
| --- | --- |
| `airline` | Airline option index |
| `source` | Source city option index |
| `destination` | Destination city option index |
| `departure` | Departure time-window option index |
| `arrival` | Arrival time-window option index |
| `flight_class` | Cabin class encoded as a number |
| `total_stops` | Number of stops |
| `duration_hours` | Flight duration in hours |
| `days_left` | Days between booking and journey |
| `journey_date` | Date in `YYYY-MM-DD` format |

Example response:

```json
{
  "prediction": 6234.5,
  "formatted_prediction": "INR 6,234.50",
  "route": "Delhi to Mumbai",
  "airline": "Indigo",
  "journey_day": "Monday",
  "confidence_note": "Estimate from the saved XGBoost model"
}
```

## Input Limits

The server validates these numeric limits before prediction:

| Input | Range |
| --- | --- |
| Duration | `0.75` to `43.58` hours |
| Days left | `1` to `50` days |
| Fare output | Clipped between `INR 1,307` and `INR 143,019` |

Source and destination must be different cities.

## Training Workflow

Use `airline.ipynb` to review or reproduce the model-building process:

1. Load and inspect the airline fare dataset.
2. Clean and preprocess the data.
3. Encode categorical features.
4. Engineer date, weekend, duration, and booking lead-time features.
5. Train and compare regression models.
6. Save the selected XGBoost model with Joblib.

## Notes

- Keep `flight_price_prediction_model.pkl` in the project root because `server.py` loads it from that location.
- The current frontend uses indexed select values that are mapped to labels in `server.py`.
- If you retrain the model with a different feature set, update the preprocessing logic in `server.py` to match the new model features.
