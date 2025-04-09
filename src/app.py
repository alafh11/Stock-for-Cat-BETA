from flask import Flask, jsonify
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
from prophet.serialize import model_from_json
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready.parquet"
MODELS_DIR = BASE_DIR / "src" / "models"


def generate_business_dates(last_date, days):
    """Generate next N business days"""
    dates = []
    current_date = last_date + timedelta(days=1)
    while len(dates) < days:
        if current_date.weekday() < 5:
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates


@app.route("/predict/prophet/<ticker>/<days>")
def predict_prophet(ticker, days):
    try:
        days = min(int(days), 30)
        model_path = MODELS_DIR / f"prophet_{ticker.lower()}.json"

        with open(model_path, "r") as fin:
            model = model_from_json(json.load(fin))

        future = model.make_future_dataframe(periods=days, freq="B")
        forecast = model.predict(future)

        return jsonify(
            {
                "ticker": ticker.upper(),
                "model": "Prophet",
                "dates": forecast.tail(days)["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "predictions": forecast.tail(days)["yhat"].tolist(),
                "confidence_intervals": forecast.tail(days)[
                    ["yhat_lower", "yhat_upper"]
                ].values.tolist(),
                "days": days,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Prophet prediction failed: {str(e)}"}), 500


@app.route("/predict/arima/<ticker>/<days>")
def predict_arima(ticker, days):
    try:
        days = min(int(days), 30)
        model_path = MODELS_DIR / f"arima_111_{ticker.lower()}.pkl"

        if not model_path.exists():
            return jsonify({"error": f"ARIMA model for {ticker} not found"}), 404

        model = joblib.load(model_path)

        df = pd.read_parquet(DATA_FILE)
        ticker_data = df[df["ticker"] == ticker.upper()]

        if len(ticker_data) < 30:
            return (
                jsonify({"error": f"Need at least 30 days of data for {ticker}"}),
                400,
            )

        forecast = model.forecast(steps=days)

        predictions = [float(x) for x in forecast]

        last_date = ticker_data.index[-1].to_pydatetime()
        future_dates = generate_business_dates(last_date, days)

        return jsonify(
            {
                "ticker": ticker.upper(),
                "model": "ARIMA",
                "dates": future_dates,
                "predictions": predictions,
                "days": days,
            }
        )

    except Exception as e:
        app.logger.error(f"ARIMA prediction error: {str(e)}")
        return jsonify({"error": f"ARIMA prediction failed: {str(e)}"}), 500


@app.route("/")
def index():
    return jsonify(
        {
            "routes": [
                "/predict/prophet/<ticker>/<days>",
                "/predict/arima/<ticker>/<days>",
            ]
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
