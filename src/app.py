from flask import Flask, jsonify
import joblib
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready.parquet"
MODELS_DIR = BASE_DIR / "src" / "models"


# Helper function for LSTM preprocessing
def preprocess_lstm_data(data, window_size=30):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    sequences = []
    for i in range(len(data_scaled) - window_size):
        sequences.append(data_scaled[i : i + window_size])
    return np.array(sequences), scaler


# Recursive multi-step LSTM prediction
def predict_lstm_multistep(model, initial_sequence, steps, scaler):
    predictions = []
    current_sequence = initial_sequence
    for _ in range(steps):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0][0])  # Extract the predicted value
        # Shift the sequence and append the new prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[pred[0]]], axis=1)
    return (
        scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        .flatten()
        .tolist()
    )


@app.route("/")
def index():
    return jsonify(
        {"routes": ["/predict/arima/<ticker>", "/predict/lstm/<ticker>/<steps>"]}
    )


@app.route("/predict/arima/<ticker>")
def predict_arima(ticker):
    try:
        ticker = ticker.upper()
        model_path = MODELS_DIR / f"arima_{ticker.lower()}.pkl"

        # Check if ARIMA model file exists
        if not model_path.exists():
            return jsonify({"error": f"ARIMA model for {ticker} not found"}), 404

        # Load ARIMA model
        model = joblib.load(model_path)

        # Generate 7-step forecast
        predictions = model.forecast(steps=7).tolist()

        # Check if predictions are valid
        if not predictions or len(predictions) == 0:
            return jsonify({"error": "ARIMA model failed to generate predictions"}), 500

        # Return predictions as JSON
        return jsonify({"ticker": ticker, "model": "ARIMA", "predictions": predictions})

    except Exception as e:
        # Return error message as JSON
        return jsonify({"error": f"ARIMA prediction failed: {str(e)}"}), 500


@app.route("/predict/lstm/<ticker>/<steps>")
def predict_lstm(ticker, steps):
    try:
        ticker = ticker.upper()
        steps = int(steps)

        # Load LSTM model and scaler
        model_path = MODELS_DIR / f"lstm_{ticker.lower()}.keras"
        scaler_path = MODELS_DIR / f"lstm_scaler.pkl"

        if not model_path.exists():
            return jsonify({"error": f"LSTM model for {ticker} not found"}), 404
        if not scaler_path.exists():
            return jsonify({"error": f"Scaler file for {ticker} not found"}), 404

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Load and preprocess data
        df = pd.read_parquet(DATA_FILE)
        data = df[df["ticker"] == ticker][["close"]].values

        if len(data) < 30:
            return (
                jsonify(
                    {"error": f"Not enough data for LSTM modeling for ticker {ticker}"}
                ),
                400,
            )

        # Prepare the initial sequence
        X, _ = preprocess_lstm_data(data, window_size=30)
        initial_sequence = X[-1:]  # Use the last sequence from data

        # Perform multi-step prediction
        predictions = predict_lstm_multistep(model, initial_sequence, steps, scaler)

        return jsonify({"ticker": ticker, "model": "LSTM", "predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
