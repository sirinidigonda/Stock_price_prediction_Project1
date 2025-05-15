Stock Price Trend Prediction with LSTM
This project predicts future stock prices using past trends by leveraging LSTM (Long Short-Term Memory), a type of recurrent neural network (RNN), for time-series forecasting. The project involves fetching stock data, training a model, and using a Streamlit dashboard to visualize predictions alongside key financial indicators such as Moving Average (MA) and Relative Strength Index (RSI).

Objective:
The primary goal of this project is to predict the future stock prices using past stock data. By applying the LSTM model on historical stock prices, we can forecast the potential price trends.

Tools & Technologies:
Python: The core programming language for implementing the model and dashboard.
Keras: Used to build and train the LSTM model.
Pandas: For data manipulation and preprocessing.
Matplotlib: To plot the graphs for visualizations.
Yahoo Finance API (yfinance): To fetch historical stock data.
Streamlit: To build and deploy an interactive dashboard for visualizing the stock predictions.

Technology Stack:
Programming Language: Python
Libraries:
yfinance: To fetch historical stock market data
pandas: For data manipulation and analysis
numpy: For numerical computations
matplotlib: For plotting graphs and visualizations
Keras: For building the LSTM model
TensorFlow: For training the model and implementing deep learning
Streamlit: For creating the interactive dashboard

Installation:
1.Clone the repository:
git clone https://github.com/sirinidigonda/Stock_price_prediction_Project1.git
cd Stock_price_prediction_Project1
2. Create a virtual environment:
python -m venv venv
3. Activate the virtual environment:
Windows:.\venv\Scripts\activate
macOS/Linux:source venv/bin/activate
4. Install dependencies:
pip install -r requirements.txt

Usage:
To run the Streamlit dashboard:
Fetch stock data using the yfinance API for a specific ticker.
Normalize and prepare the data by converting it into a form that can be used by the LSTM model.
Train and validate the LSTM model on the data.
Plot the predictions vs. actual stock prices and visualize key financial indicators.
Optionally, deploy the dashboard to Streamlit for user interaction.
Run the following command to start the Streamlit app:
streamlit run stock_price_dashboard.py
The app will start locally and can be accessed at: http://localhost:8501

Steps:
Data Collection: Stock price data is fetched using the Yahoo Finance API via the yfinance library.
Data Preprocessing: The data is cleaned and normalized to make it suitable for training the LSTM model.
Building the LSTM Model:
The LSTM model is built using Keras.
The model takes the historical stock price data as input and predicts the future stock prices.
Model Training and Validation: The model is trained and validated using historical stock data.
Predictions vs Actual: The predicted stock prices are compared with actual values and visualized.
Visualization:
Moving Average (MA): A 20-day moving average is calculated and plotted alongside stock prices.
RSI (Relative Strength Index): The RSI indicator is plotted to visualize whether a stock iserbought or oversold.

Visualizations:
Stock Price vs 20-Day Moving Average (MA20):
A line chart showing the stock's actual prices and its 20-day moving average.
Relative Strength Index (RSI):
A chart showing the RSI of the stock, indicating overbought or oversold conditions.
Predictions vs Actual:
The stock's predicted prices are compared with the actual closing prices in a plot.

Dependencies:
streamlit
pandas
numpy
yfinance

Deployment:
This project can be deployed using Streamlit Cloud or Heroku for online access. You can also run it locally using Streamlit.

Demo: 
Screen recording available in the repository
Streamlit link is also provided in below section

Deliverables:
VS Code Project: A Python-based project developed in VS Code that demonstrates the complete workflow — data preparation, model building, training, validation, and prediction.
Model Weights: The trained LSTM model weights saved in .h5 format.
Graphs: Visualizations for stock price predictions, 20-Day Moving Average, and RSI (Relative Strength Index).
Streamlit Link: The deployed Streamlit dashboard for user interaction. Here is the link:https://stockpricepredictionproject1-ahorgzdnl3li59v3rp9nqc.streamlit.app/

Future Improvements:
Hyperparameter Tuning: Optimizing the model for better performance.
Advanced Models: Experimenting with more advanced models like GRU (Gated Recurrent Unit) for improved predictions.
Additional Indicators: Incorporating more technical indicators such as Bollinger Bands or MACD for enhanced analysis.



