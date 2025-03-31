import streamlit as st
import requests
import pandas as pd

st.title("AAPL Stock Forecaster")
model_type = st.selectbox("Model", ["ARIMA", "LSTM"])

try:
    response = requests.get(f"http://localhost:5000/predict/{model_type.lower()}")
    if response.status_code == 200:
        data = response.json()
        st.line_chart(pd.DataFrame(data["predictions"], columns=["Price"]))
    else:
        st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
except Exception as e:
    st.error(f"Connection failed: {str(e)}")
#
