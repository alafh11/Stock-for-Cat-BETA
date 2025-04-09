# Stock For Cat (BETA) 📈🐱
## 📌 Project Overview


This is a beta version of a stock analysis dashboard that currently focuses on AAPL (Apple Inc.) stock as a proof of concept. The project demonstrates a scalable pipeline for financial data analysis and forecasting that can be expanded to other stocks in the future.

## 🚀 Key Features

Proof of Concept – Currently supports AAPL (1/100 planned stocks)

End-to-End Analysis Pipeline:

Data importation from Yahoo Finance

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Forecasting using Prophet and ARIMA models (experimental, see Model Status below)
Interactive Dashboard: Basic Streamlit app for visualizing forecasts

## 🔮 Future Plans

### 📈 Additional Models

### 📊 Multi-Stock Expansion

### 🎛️ Dashboard Upgrades


## ⚠️ Model Status
⚠️ Experimental Phase – Current models (Prophet & ARIMA) are under evaluation and may require further tuning for optimal performance. Contributions and feedback are welcome!


## 🛠️ Installation

### 1 Clone and Setup the Project:
```
git clone https://github.com/alafh11/Stock-for-Cat-BETA.git
cd Stock-for-Cat-BETA
pip install -r requirements.txt
```
### 2  Launch the Application:

You'll need to run two services in separate terminals:

🖥️ Terminal 1: Flask Backend Server
```
py .\app.py
```

📊 Terminal 2: Streamlit Dashboard
```
streamlit run dashboard.py
```




