
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Function to compute RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit app title
st.title("📈 Stock Price Prediction Dashboard")

# Input fields
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# Fetch data and proceed if ticker is entered
if ticker:
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, group_by='column', auto_adjust=False)

    if data.empty:
        st.error("❌ Failed to fetch data. Please check the ticker and date range.")
        st.stop()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join([str(c) for c in col]).strip() for col in data.columns.values]

    # Show available columns
    st.write("✅ Available columns:", list(data.columns))

    # Identify the close price column
    close_candidates = [col for col in data.columns if 'Close' in col]
    if not close_candidates:
        st.error("❌ No 'Close' column found in the data.")
        st.stop()
    price_col = close_candidates[0]

    # Prepare DataFrame
    data = data[[price_col]].dropna()
    data['MA20'] = data[price_col].rolling(window=20).mean()
    data['RSI'] = compute_rsi(data[price_col])

    # Plot moving average chart
    st.subheader("📊 Stock Price & 20-Day Moving Average")
    st.line_chart(data[[price_col, 'MA20']])

    # Plot RSI chart
    st.subheader("📉 Relative Strength Index (RSI)")
    st.line_chart(data['RSI'])

    # Show data table
    st.subheader("📋 Recent Data")
    st.dataframe(data.tail())



import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Function to compute RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit app title
st.title("📈 Stock Price Prediction Dashboard")

# Input fields
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# Fetch data and proceed if ticker is entered
if ticker:
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, group_by='column', auto_adjust=False)

    if data.empty:
        st.error("❌ Failed to fetch data. Please check the ticker and date range.")
        st.stop()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join([str(c) for c in col]).strip() for col in data.columns.values]

    # Show available columns
    st.write("✅ Available columns:", list(data.columns))

    # Identify the close price column
    close_candidates = [col for col in data.columns if 'Close' in col]
    if not close_candidates:
        st.error("❌ No 'Close' column found in the data.")
        st.stop()
    price_col = close_candidates[0]

    # Prepare DataFrame
    data = data[[price_col]].dropna()
    data['MA20'] = data[price_col].rolling(window=20).mean()
    data['RSI'] = compute_rsi(data[price_col])

    # Plot moving average chart
    st.subheader("📊 Stock Price & 20-Day Moving Average")
    st.line_chart(data[[price_col, 'MA20']])

    # Plot RSI chart
    st.subheader("📉 Relative Strength Index (RSI)")
    st.line_chart(data['RSI'])

    # Show data table
    st.subheader("📋 Recent Data")
    st.dataframe(data.tail())

