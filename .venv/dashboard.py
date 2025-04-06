import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(page_title="Stock Forecaster Pro", layout="wide")
st.title("Stock For Cat 🐱 ")
st.markdown("Predict future prices using Prophet or LSTM models")

with st.sidebar:
    st.header("Model Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
    model_type = st.selectbox("Forecasting Model", ["Prophet", "LSTM"])
    days = st.slider("Forecast Horizon (days)", 1, 30, 7)

    if model_type == "Prophet":
        st.info("Prophet: Best for trends and seasonality")
    else:
        st.info("LSTM: Captures complex patterns")

if st.button("Generate Forecast", type="primary"):
    try:
        endpoint = f"http://localhost:5000/predict/{model_type.lower()}/{ticker}/{days}"

        with st.spinner(f"Generating {days}-day {model_type} forecast..."):
            response = requests.get(endpoint)

            if response.status_code == 200:
                data = response.json()
                
                forecast_df = pd.DataFrame({
                    "Date": pd.to_datetime(data["dates"]),
                    "Predicted Price": data["predictions"],
                })

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    forecast_df["Date"],
                    forecast_df["Predicted Price"],
                    marker="o",
                    linestyle="--",
                    label="Prediction",
                )

                if "confidence_intervals" in data:
                    conf_int = np.array(data["confidence_intervals"])
                    ax.fill_between(
                        forecast_df["Date"],
                        conf_int[:, 0],
                        conf_int[:, 1],
                        color="gray",
                        alpha=0.2,
                        label="95% Confidence",
                    )

                ax.set_title(f"{ticker} {model_type} Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                display_df = forecast_df.copy()
                display_df["Date"] = display_df["Date"].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    display_df.style.format({
                        "Predicted Price": "${:.2f}",
                        "Date": "{}",
                    }),
                    hide_index=True,
                )

            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to prediction server. Is Flask app running?")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

with st.expander("Model Details"):
    if model_type == "Prophet":
        st.markdown(
            """
            **Prophet Model**  
            • Automatic trend detection  
            • Handles holidays and seasonality  
            • Provides confidence intervals  
            • Best for medium-term forecasts  
            """
        )
    else:
        st.markdown(
            """
            **LSTM Model**  
            • Deep learning approach  
            • Captures non-linear patterns  
            • Requires more historical data  
            • Better for volatile stocks  
            """
        )

st.markdown("---")
st.caption("Stock Forecaster Pro | Supports all major US tickers")