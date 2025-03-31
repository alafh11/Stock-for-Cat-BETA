import streamlit as st
import requests
import pandas as pd

st.title("Stock Price Forecaster")
st.write("Predict stock prices using ARIMA or LSTM models.")

# Input stock ticker and model type
ticker = st.text_input("Enter Stock Ticker", "AAPL")
model_type = st.selectbox("Choose Model", ["ARIMA", "LSTM"])

# Only show steps slider for LSTM
if model_type == "LSTM":
    steps = st.slider("Prediction Steps (Future Days)", 1, 30, 7)
else:
    steps = 7  # ARIMA uses fixed 7 steps in your API

if st.button("Get Predictions"):
    try:
        # Build different endpoints for ARIMA and LSTM
        if model_type == "ARIMA":
            endpoint = f"http://localhost:5000/predict/arima/{ticker.lower()}"
        else:
            endpoint = f"http://localhost:5000/predict/lstm/{ticker.lower()}/{steps}"

        response = requests.get(endpoint)

        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]

            if predictions:
                st.write(
                    f"Predictions for {ticker.upper()} using {model_type.upper()} (Next {len(predictions)} Days):"
                )
                st.line_chart(pd.DataFrame(predictions, columns=["Price"]))
            else:
                st.warning(
                    f"No predictions returned for {ticker.upper()} using {model_type.upper()}."
                )
        else:
            error_message = response.json().get("error", "Unknown error")
            st.error(f"API Error: {error_message}")
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
