from flask import Flask, jsonify
import joblib
import tensorflow as tf
from pathlib import Path
import os

app = Flask(__name__)

# Set up correct paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


@app.route("/predict/<model_type>")
def predict(model_type):
    try:
        if model_type == "arima":
            model = joblib.load(MODELS_DIR / "arima_aapl.pkl")
            pred = model.forecast(steps=7).tolist()
        elif model_type == "lstm":
            model = tf.keras.models.load_model(MODELS_DIR / "lstm_aapl.keras")
            # Add your LSTM prediction logic here
            pred = [100, 101, 102]  # Placeholder - replace with real predictions
        else:
            return jsonify({"error": "Invalid model type"}), 400

        return jsonify({"predictions": pred, "model": model_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
