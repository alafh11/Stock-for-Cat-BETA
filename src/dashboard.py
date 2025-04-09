import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Stock Forecaster Pro", layout="wide")
st.title("📈 Stock Forecaster Pro")
st.markdown("Predict future prices using advanced forecasting models")

with st.sidebar:
    st.header("Model Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
    model_type = st.selectbox("Forecasting Model", ["Prophet", "ARIMA"])
    days = st.slider("Forecast Horizon (days)", 1, 30, 7)

    st.markdown("---")
    st.markdown("**Model Recommendations:**")
    if model_type == "Prophet":
        st.info("✅ Best for trends and seasonality")
    else:
        st.info("✅ Best for short-term forecasts")


def plot_forecast(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        pd.to_datetime(data["dates"]),
        data["predictions"],
        marker="o",
        linestyle="--",
        label="Prediction",
    )

    if "confidence_intervals" in data:
        conf_int = np.array(data["confidence_intervals"])
        ax.fill_between(
            pd.to_datetime(data["dates"]),
            conf_int[:, 0],
            conf_int[:, 1],
            color="gray",
            alpha=0.2,
            label="95% Confidence",
        )

    ax.set_title(f"{data['ticker']} {data['model']} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


if st.button("Generate Forecast", type="primary"):
    endpoint = f"http://localhost:5000/predict/{model_type.lower()}/{ticker}/{days}"
    with st.spinner(f"Generating {days}-day {model_type} forecast..."):
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()

            plot_forecast(data)

            df = pd.DataFrame(
                {
                    "Date": data["dates"],
                    "Predicted Price": data["predictions"],
                }
            )

            if "confidence_intervals" in data:
                df["Lower Bound"] = np.array(data["confidence_intervals"])[:, 0]
                df["Upper Bound"] = np.array(data["confidence_intervals"])[:, 1]

            st.dataframe(
                df.style.format(
                    {
                        "Predicted Price": "${:.2f}",
                        "Lower Bound": "${:.2f}",
                        "Upper Bound": "${:.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to prediction server. Is Flask app running?")
        except ValueError as json_err:
            st.error(f"Failed to parse response: {json_err}")
            st.text(f"Raw response: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


with st.expander("📚 Model Documentation"):
    if model_type == "Prophet":
        st.markdown(
            """
            **Prophet Model**  
            - Developed by Facebook  
            - Handles seasonality and holidays automatically  
            - Provides uncertainty intervals  
            - Best for 1 week to 6 month forecasts  
            """
        )
    else:
        st.markdown(
            """
            **ARIMA Model**  
            - Classical time series approach  
            - Excellent for short-term forecasts  
            - Fast prediction times  
            - Works best with stationary data  
            """
        )

st.markdown("---")
st.caption("© 2025 Stock Forecaster Pro | Data updates daily")
