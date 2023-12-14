
import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

st.title('Stock-trend-Predition')
user_input = st.text_input('Enter stock Ticker','BSE/BOM500325')
# Set your Quandl API key
quandl.ApiConfig.api_key = '7BZmWnxjHb4zoXMfC4vX'

# Function to fetch historical data for n years
def fetch_historical_data(symbol):
    try:
        # Get historical stock data for the symbol (e.g., 'BSE/BOM500325' for Reliance Industries Limited)
        today = pd.Timestamp.now()
        three_years_ago = today - pd.DateOffset(years=8)
        data = quandl.get(symbol, start_date=three_years_ago, end_date=today)
        return data

    except Exception as e:
        print("Error fetching historical data:", e)
        return None

# Function to fetch intraday data for the last 30 days
def fetch_intraday_data(symbol):
    try:
        # Get intraday stock data for the symbol (e.g., 'BSE/BOM500325' for Reliance Industries Limited)
        today = pd.Timestamp.now()
        thirty_days_ago = today - pd.DateOffset(days=30)
        data = quandl.get(symbol, start_date=thirty_days_ago, end_date=today, collapse='daily')
        return data

    except Exception as e:
        print("Error fetching intraday data:", e)
        return None

# Example: Fetch historical data for Reliance Industries Limited (BSE code: 500325)
historical_data = fetch_historical_data(user_input)  # Replace with desired BSE code

# Example: Fetch intraday data for the last 30 days for Reliance Industries Limited (BSE code: 500325)
intraday_data = fetch_intraday_data(user_input)  # Replace with desired BSE code

if historical_data is not None and intraday_data is not None:
    # Concatenate historical and intraday data into a single dataframe
    combined_data = pd.concat([historical_data, intraday_data])

data = combined_data
data = data.reset_index()


st.subheader('Data from last 10 years')
st.write(data.describe())

st.subheader('Closing price vs Timechart')
# Create a Plotly figure for closing prices
fig = go.Figure()

# Add trace for closing prices
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))

# Update layout and display the plot
fig.update_layout(title='Stock Closing Prices',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_white',
                 height=600,  # Set the height of the figure
                  width=1000)  # You can use different templates like 'plotly_dark', 'plotly' etc.

st.plotly_chart(fig)

st.subheader('100 and 200 days Moving average')
# Calculate 100 and 200 days moving averages
data['100_MA'] = data['Close'].rolling(window=100).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
# Create a Plotly figure for closing prices and moving averages
fig = go.Figure()

# Add traces for closing prices and moving averages
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['100_MA'], mode='lines', name='100 Days Moving Average'))
fig.add_trace(go.Scatter(x=data.index, y=data['200_MA'], mode='lines', name='200 Days Moving Average'))

# Update layout and display the plot
fig.update_layout(title='Stock Closing Prices and Moving Averages',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_white',
                  height=600,  # Set the height of the figure
                  width=1200)  # Set the width of the figure

st.plotly_chart(fig)

closing_prices = data['Close'].values.reshape(-1, 1)

# Scale the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(closing_prices)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (e.g., using closing prices of 30 days to predict the 31st day)
sequence_length = 30
X, y = create_sequences(closing_prices_scaled, sequence_length)

train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
test_X =  X[train_size:]
test_y =  y[train_size:]

from tensorflow.keras.models import load_model
model = load_model('keras-model.h5')

predicted_stock_prices = model.predict(test_X)
# Inverse transform the scaled predictions to original values
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)
test_y = scaler.inverse_transform(test_y)

st.subheader('Predicted vs actual prices')
fig2 = plt.figure(figsize=(15, 8))
plt.plot(data.index[-len(closing_prices):], closing_prices, color='blue', label='Actual Closing Prices')
plt.plot(data.index[-len(predicted_stock_prices):], predicted_stock_prices, color='red', label='Predicted Closing Prices')
plt.title('Actual vs. Predicted Stock Closing Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)

