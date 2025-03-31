import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure page
st.set_page_config(page_title="Stock Forecaster Pro", layout="wide")
st.title("📊 Stock Price Forecaster")
st.markdown("Predict future prices using ARIMA or LSTM models (1-30 days forecast)")

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")

    model_type = st.radio("Forecasting Model", ["ARIMA", "LSTM"], index=0)

    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()

    steps = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Select number of days to predict (1-30)",
    )

    st.markdown("---")
    st.info("Note: Confidence intervals show prediction uncertainty")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Generate Forecast", type="primary"):
        try:
            endpoint = f"http://localhost:5000/predict/{model_type.lower()}/{ticker.lower()}/{steps}"

            with st.spinner(f"Generating {steps}-day forecast..."):
                response = requests.get(endpoint)

                if response.status_code == 200:
                    data = response.json()

                    forecast_df = pd.DataFrame(
                        {
                            "Day": np.arange(1, steps + 1),
                            "Predicted Price": data["predictions"],
                        }
                    )

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(
                        forecast_df["Day"],
                        forecast_df["Predicted Price"],
                        marker="o",
                        linestyle="--",
                        color="#1f77b4",
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

                    ax.set_title(
                        f"{ticker} {model_type} {steps}-Day Forecast\n", pad=20
                    )
                    ax.set_xlabel("Days in Future")
                    ax.set_ylabel("Price ($)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    st.pyplot(fig)
                    st.dataframe(
                        forecast_df.style.format({"Predicted Price": "${:.2f}"})
                    )

                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection failed - ensure the Flask server is running")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.markdown("### Model Information")

    if model_type == "ARIMA":
        st.markdown(
            """
        **ARIMA Model**  
        • Statistical time series model  
        • Provides confidence intervals  
        • Best for short-medium term  
        • Requires stationary data  
        """
        )
    else:
        st.markdown(
            """
        **LSTM Model**  
        • Deep learning approach  
        • Captures complex patterns  
        • Needs more historical data  
        • Better for volatile stocks  
        """
        )

    st.markdown("---")
    st.markdown(
        """
    **How to Use**  
    1. Enter stock ticker  
    2. Select model type  
    3. Choose forecast days  
    4. Click Generate  
    """
    )

st.markdown("---")
st.caption("Stock Forecaster Pro | Supports all major US tickers")
