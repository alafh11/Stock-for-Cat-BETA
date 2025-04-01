import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Stock Forecaster Pro", layout="wide")
st.title("📊 Advanced Stock Forecaster")
st.markdown("Predict future prices using Prophet, ARIMA or LSTM models")

with st.sidebar:
    st.header("Model Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
    model_type = st.selectbox("Forecasting Model", ["Prophet", "ARIMA", "LSTM"])
    days = st.slider("Forecast Horizon (days)", 1, 30, 7)

    if model_type == "Prophet":
        st.info("Prophet: Best for trends and seasonality")
    elif model_type == "ARIMA":
        st.warning("ARIMA: Works best with stationary data")
    else:
        st.info("LSTM: Captures complex patterns")

if st.button("Generate Forecast", type="primary"):
    try:
        endpoint = f"http://localhost:5000/predict/{model_type.lower()}/{ticker}/{days}"

        with st.spinner(f"Generating {days}-day {model_type} forecast..."):
            response = requests.get(endpoint)

            if response.status_code == 200:
                data = response.json()

                forecast_df = pd.DataFrame(
                    {
                        "Day": np.arange(1, days + 1),
                        "Predicted Price": data["predictions"],
                    }
                )

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    forecast_df["Day"],
                    forecast_df["Predicted Price"],
                    marker="o",
                    linestyle="--",
                    label="Prediction",
                )

                if "confidence_intervals" in data:
                    conf_int = np.array(data["confidence_intervals"])
                    ax.fill_between(
                        forecast_df["Day"],
                        conf_int[:, 0],
                        conf_int[:, 1],
                        color="gray",
                        alpha=0.2,
                        label="95% Confidence",
                    )

                ax.set_title(f"{ticker} {model_type} Forecast")
                ax.set_xlabel("Days Ahead")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.dataframe(forecast_df.style.format({"Predicted Price": "${:.2f}"}))

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
    elif model_type == "ARIMA":
        st.markdown(
            """
        **ARIMA Model**  
        • Statistical time series model  
        • Works best with stationary data  
        • Good for short-term forecasts  
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
