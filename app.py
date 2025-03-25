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

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Drop the Date column as it's not needed for prediction
df.drop(columns=['Date'], inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=["Stock Price"])
y = df["Stock Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

def plot_results():
    # Visualizing actual vs predicted stock prices
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label="Perfect Fit")
    ax1.set_xlabel("Actual Stock Price")
    ax1.set_ylabel("Predicted Stock Price")
    ax1.set_title("Actual vs Predicted Stock Prices")
    ax1.legend()
    
    # Plot residuals
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.histplot(residuals, bins=30, kde=True, ax=ax2)
    ax2.set_xlabel("Residual Error")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Error Distribution")
    
    return fig1, fig2

# Streamlit UI
st.title("Stock Price Prediction using Multiple Linear Regression")
st.write("### Click the button below to generate the prediction graphs.")

if st.button("Generate Graphs"):
    fig1, fig2 = plot_results()
    st.pyplot(fig1)
    st.pyplot(fig2)
    
    st.write("### Model Performance Metrics:")
    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
    st.write(f"**R-squared (R2):** {r2}")


