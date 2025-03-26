import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("synthetic_stock_data.csv")

# Convert 'Date' column to datetime format and drop it
df['Date'] = pd.to_datetime(df['Date'])
df.drop(columns=['Date'], inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=["Stock Price"])
y = df["Stock Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Stock Price Prediction using Multiple Linear Regression")

# Sliders for user input
st.write("### Input values for prediction:")
interest_rate = st.slider("Interest Rate (%)", float(X["Interest Rate (%)"].min()), float(X["Interest Rate (%)"].max()), float(X["Interest Rate (%)"].mean()))
inflation = st.slider("Inflation Rate (%)", float(X["Inflation Rate (%)"].min()), float(X["Inflation Rate (%)"].max()), float(X["Inflation Rate (%)"].mean()))
gdp_growth = st.slider("GDP Growth Rate (%)", float(X["GDP Growth Rate (%)"].min()), float(X["GDP Growth Rate (%)"].max()), float(X["GDP Growth Rate (%)"].mean()))
market_index = st.slider("Market Index", float(X["Market Index"].min()), float(X["Market Index"].max()), float(X["Market Index"].mean()))

# Store user inputs
user_input = np.array([[interest_rate, inflation, gdp_growth, market_index]])

# Button to generate graph
if st.button("Generate Graph"):
    # Predict stock price based on user input
    predicted_price = model.predict(user_input)[0]

    # Plot the regression graph
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, model.predict(X_test), alpha=0.5, label="Model Predictions")
    ax.scatter(predicted_price, predicted_price, color='red', marker='o', s=100, label="User Input Prediction")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='dashed', label="Perfect Fit")
    ax.set_xlabel("Actual Stock Price")
    ax.set_ylabel("Predicted Stock Price")
    ax.set_title("Stock Price Prediction Based on User Input")
    ax.legend()

    # Display the graph
    st.pyplot(fig)

    # Display predicted stock price
    st.write(f"### Predicted Stock Price: **{predicted_price:.2f}**")







#https://stock-graph.streamlit.app/

