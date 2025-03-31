from flask import Flask, jsonify
import joblib
import tensorflow as tf
from pathlib import Path
import os

app = Flask(__name__)
MODELS_DIR = Path("models")


@app.route("/predict/<ticker>/<model_type>")
def predict(ticker, model_type):
    try:
        if model_type == "arima":
            if not (MODELS_DIR / "arima_aapl.pkl").exists():
                return jsonify({"error": "ARIMA model not found"}), 404
            model = joblib.load(MODELS_DIR / "arima_aapl.pkl")
            pred = model.forecast(steps=7).tolist()
        elif model_type == "lstm":
            model = tf.keras.models.load_model(MODELS_DIR / "lstm_aapl.keras")
            pred = model.predict().tolist()
        else:
            return jsonify({"error": "Invalid model type"}), 400

        return jsonify({"predictions": pred, "model": model_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
