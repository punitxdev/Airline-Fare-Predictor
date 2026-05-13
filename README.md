# Airline Fare Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-blue.svg)](https://pandas.pydata.org/)

## Project Overview

The Airline Fare Predictor is a machine learning web application designed to forecast Indian domestic flight fares. It leverages a trained XGBoost regression model exposed via a Flask backend, and a responsive web interface for seamless user interaction. The model considers various parameters including route, airline, cabin class, layovers, departure and arrival time windows, journey date, travel duration, and booking lead time to provide accurate fare estimations.

## Key Features

- **Real-time Fare Prediction:** Provides instant flight fare estimations in INR utilizing a robust XGBoost regression model.
- **Interactive Web Interface:** A browser-based booking-console style frontend with live route summaries and form validation.
- **RESTful API:** Exposes a Flask `/predict` endpoint for model inference, allowing for easy integration with other services.
- **Comprehensive Validation:** Server-side input validation for city routes, flight durations, and booking lead times to ensure model reliability.
- **End-to-End ML Pipeline:** Includes Jupyter notebooks documenting the entire process from data preprocessing and exploratory data analysis to model training, evaluation, and export.

## Technical Architecture

The application is built using a modern, lightweight technology stack:

- **Backend:** Python, Flask
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Joblib
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

## Dataset Description

The application utilizes a comprehensive dataset of flight records. The cleaned dataset (`Cleaned_dataset.csv`) contains engineered features crucial for accurate predictions:

| Feature | Description |
|---------|-------------|
| `Airline` | Operating airline (e.g., Indigo, Air India, Vistara) |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Departure` | Departure time window |
| `Arrival` | Arrival time window |
| `Class` | Cabin class |
| `Total_stops` | Number of layovers |
| `Duration_in_hours` | Total flight duration in hours |
| `Days_left` | Number of days between booking and journey |
| `Fare` | Flight fare in INR (Target Variable) |

Currently supported cities: Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai.

## Model Performance

Extensive comparative analysis was conducted across multiple regression algorithms, including Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Multi-Layer Perceptron (MLP).

The deployed XGBoost Regressor (`flight_price_prediction_model.pkl`) achieved an approximate **95.58% R-squared (R2) score** on the holdout test set, demonstrating high accuracy in predicting fare variations.

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/punitxdev/Airline-Fare-Predictor.git
   cd Airline-Fare-Predictor
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python server.py
   ```

5. **Access the web interface**
   Navigate to `http://127.0.0.1:5000` in your web browser.

## API Reference

The application provides a RESTful endpoint for fare predictions.

### Endpoint: `/predict`
- **Method:** `POST`
- **Content-Type:** `application/x-www-form-urlencoded`

**Request Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `airline` | Integer | Airline option index |
| `source` | Integer | Source city option index |
| `destination` | Integer | Destination city option index |
| `departure` | Integer | Departure time-window option index |
| `arrival` | Integer | Arrival time-window option index |
| `flight_class` | Integer | Cabin class (encoded) |
| `total_stops` | Integer | Number of layovers |
| `duration_hours` | Float | Flight duration in hours |
| `days_left` | Integer | Days until the journey |
| `journey_date` | Date | Date in `YYYY-MM-DD` format |

**Successful Response:**
```json
{
  "prediction": 6234.5,
  "formatted_prediction": "₹6,234.50",
  "route": "Delhi to Mumbai",
  "airline": "Indigo",
  "journey_day": "Monday",
  "confidence_note": "Estimate from the saved XGBoost model"
}
```

## Project Structure

```text
Airline-Fare-Predictor/
├── airline.ipynb                         # Data analysis and model training
├── Cleaned_dataset.csv                   # Cleaned training dataset
├── Scraped_dataset.csv                   # Original scraped dataset
├── flight_price_prediction_model.pkl     # Serialized XGBoost model
├── index.html                            # Frontend user interface
├── styles.css                            # Frontend styling
├── server.py                             # Flask application backend
├── requirements.txt                      # Project dependencies
└── archive.zip                           # Archived source data
```
