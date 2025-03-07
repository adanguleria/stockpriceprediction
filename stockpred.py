import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Title of the app
st.title("Advanced Stock Price Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
prediction_days = st.sidebar.slider("Number of Days to Predict:", 1, 30, 10)
model_choice = st.sidebar.selectbox("Select Model:", ["LSTM", "ARIMA"])

# Fetch stock data and company name
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        company_name = stock.info.get('longName', 'N/A')  # Get company name
        return data, company_name
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), "N/A"

data, company_name = load_data(ticker, start_date, end_date)

# Display company name
st.subheader(f"Company: {company_name}")

# Display raw data
st.subheader("Raw Stock Data")
st.write(data.tail())

# Plot closing price
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Preprocess data for LSTM
def preprocess_data(data, prediction_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ARIMA Model
def build_arima_model(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

# Predict future prices
if model_choice == "LSTM":
    st.subheader("LSTM Model Prediction")
    x_train, y_train, scaler = preprocess_data(data, prediction_days)
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Predict future prices
    test_data = data['Close'].values
    test_data = scaler.transform(test_data.reshape(-1, 1))
    x_test = []
    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[i-prediction_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index[prediction_days:], data['Close'][prediction_days:], label='Actual Price')
    ax.plot(data.index[prediction_days:], predicted_prices, label='Predicted Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

elif model_choice == "ARIMA":
    st.subheader("ARIMA Model Prediction")
    model = build_arima_model(data)
    forecast = model.forecast(steps=prediction_days)
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Actual Price')
    ax.plot(future_dates, forecast, label='Predicted Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

# Display predicted prices
st.subheader("Predicted Prices")
if model_choice == "LSTM":
    st.write(pd.DataFrame({
        "Date": data.index[prediction_days:],
        "Predicted Price": predicted_prices.flatten()
    }))
elif model_choice == "ARIMA":
    st.write(pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": forecast
    }))

# New Feature: Top Stocks of the Day
st.subheader("Top Stocks of the Day")

# Fetch top gainers, losers, and most active stocks
@st.cache_data
def get_top_stocks():
    # Fetch top gainers
    gainers = yf.Tickers("AAPL MSFT GOOGL AMZN TSLA").history(period="1d")
    gainers = gainers['Close'].pct_change().dropna().sort_values(ascending=False).head(5)

    # Fetch top losers
    losers = yf.Tickers("AAPL MSFT GOOGL AMZN TSLA").history(period="1d")
    losers = losers['Close'].pct_change().dropna().sort_values(ascending=True).head(5)

    # Fetch most active stocks (by volume)
    active = yf.Tickers("AAPL MSFT GOOGL AMZN TSLA").history(period="1d")
    active = active['Volume'].sort_values(ascending=False).head(5)

    return gainers, losers, active

gainers, losers, active = get_top_stocks()

# Display top gainers
st.write("### Top Gainers")
st.write(gainers)

# Display top losers
st.write("### Top Losers")
st.write(losers)

# Display most active stocks
st.write("### Most Active Stocks")
st.write(active)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")