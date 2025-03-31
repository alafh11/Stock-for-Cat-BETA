from flask import Flask, jsonify
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model  # Added this import
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready.parquet"
MODELS_DIR = BASE_DIR / "src" / "models"


@app.route("/predict/arima/<ticker>/<steps>")
def predict_arima(ticker, steps):
    try:
        ticker = ticker.upper()
        steps = min(int(steps), 30)  # Keep max 30 days

        model_path = MODELS_DIR / f"arima_{ticker.lower()}.pkl"

        if not model_path.exists():
            return jsonify({"error": f"ARIMA model for {ticker} not found"}), 404

        # Load pmdarima model
        model = joblib.load(model_path)

        # Get predictions (pmdarima syntax)
        predictions, conf_int = model.predict(n_periods=steps, return_conf_int=True)

        return jsonify(
            {
                "ticker": ticker,
                "model": "ARIMA",
                "predictions": predictions.tolist(),
                "confidence_intervals": conf_int.tolist(),
                "steps": steps,
            }
        )

    except Exception as e:
        return jsonify({"error": f"ARIMA prediction failed: {str(e)}"}), 500


@app.route("/predict/lstm/<ticker>/<steps>")
def predict_lstm(ticker, steps):
    try:
        ticker = ticker.upper()
        steps = min(int(steps), 30)

        # Model paths
        model_path = MODELS_DIR / f"lstm_{ticker.lower()}.keras"
        scaler_path = MODELS_DIR / f"lstm_scaler.pkl"

        if not model_path.exists():
            return jsonify({"error": f"LSTM model for {ticker} not found"}), 404
        if not scaler_path.exists():
            return jsonify({"error": f"Scaler for {ticker} not found"}), 404

        # Load artifacts
        model = load_model(model_path)  # Now properly imported
        scaler = joblib.load(scaler_path)

        # Prepare data
        df = pd.read_parquet(DATA_FILE)
        data = df[df["ticker"] == ticker][["close"]].values

        if len(data) < 30:
            return (
                jsonify({"error": f"Need at least 30 days of data for {ticker}"}),
                400,
            )

        # Scale data
        data_scaled = scaler.transform(data)

        # Create sequence
        sequence = data_scaled[-30:].reshape(1, 30, 1)

        # Generate predictions
        predictions = []
        current_seq = sequence.copy()

        for _ in range(steps):
            pred = model.predict(current_seq, verbose=0)[0][0]
            predictions.append(pred)
            # Update sequence
            current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

        # Inverse transform to original scale
        predictions = (
            scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            .flatten()
            .tolist()
        )

        return jsonify(
            {
                "ticker": ticker,
                "model": "LSTM",
                "predictions": predictions,
                "steps": steps,
            }
        )

    except Exception as e:
        return jsonify({"error": f"LSTM prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
