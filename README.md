# Stock For Cat (BETA) 📈🐱
## 📌 Project Overview


This is a beta version of a stock analysis dashboard that currently focuses on AAPL (Apple Inc.) stock as a proof of concept. The project demonstrates a pipeline for financial data analysis and forecasting that can be expanded to other stocks in the future.

## 🚀 Key Features

Proof of Concept: Currently tested on AAPL stock (1/100 planned)

Complete Analysis Pipeline:

Data importation from Yahoo Finance

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Forecasting with Prophet and LSTM models

Interactive Dashboard: Basic Streamlit app for visualizing forecasts

## 🔮 Future Plans

### 📈 Additional Models

### 📊 Multi-Stock Expansion

### 🎛️ Dashboard Upgrades


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
python src/app.py
```

📊 Terminal 2: Streamlit Dashboard
```
streamlit run src/dashboard.py
```




