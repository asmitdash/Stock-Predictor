
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# --- Function to calculate Market Index based on parameters ---
def calculate_market_index(interest_rate, inflation_rate, gdp_growth):
    # Define weight factors
    interest_impact = -1.5
    inflation_impact = -1.2
    gdp_impact = 2.0
    
    # Linear formula for market index
    market_index = 100 + (interest_impact * interest_rate) + (inflation_impact * inflation_rate) + (gdp_impact * gdp_growth)
    return max(50, min(market_index, 200))  # Limit values to a reasonable range

# --- Streamlit Interface ---
st.set_page_config(page_title="Market Index Simulator", layout="centered")
st.title("Dynamic Market Index Prediction")
st.markdown("Adjust the sliders to simulate how changes in **Interest Rate**, **Inflation**, and **GDP Growth** affect the Market Index.")

# --- Sliders for user input ---
interest_rate = st.slider("Interest Rate (%)", 0.0, 15.0, 5.0, 0.1)
inflation_rate = st.slider("Inflation Rate (%)", 0.0, 15.0, 5.0, 0.1)
gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 2.0, 0.1)

# --- Button to generate graphs ---
if st.button("Generate Graph"):
    # 1. Calculate Market Index
    market_index = calculate_market_index(interest_rate, inflation_rate, gdp_growth)
    st.subheader(f"Predicted Market Index: {market_index:.2f}")

    # 2. Bar Graph of Market Index
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.bar(["Market Index"], [market_index], color='skyblue')
    ax1.set_ylim(50, 200)
    ax1.set_ylabel("Market Index Value")
    ax1.set_title("Market Index Based on Economic Indicators")
    st.pyplot(fig1)

    # 3. Scatter Plot (Random samples with current prediction)
    np.random.seed(42)
    actual_values = np.random.uniform(100, 200, 100)
    noise = np.random.normal(0, 5, size=100)
    predicted_values = actual_values + noise

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.scatter(actual_values, predicted_values, label="Sample Predictions", alpha=0.6)
    ax2.plot([100, 200], [100, 200], linestyle='--', color='blue', label='Perfect Fit')
    ax2.scatter(market_index, market_index, color='red', label="Current Prediction", s=100)
    ax2.set_xlabel("Actual Stock Price")
    ax2.set_ylabel("Predicted Stock Price")
    ax2.set_title("Prediction Comparison")
    ax2.legend()
    st.pyplot(fig2)

    # 4. Line Graph for parameter sensitivity
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    x_vals = np.linspace(0, 15, 100)

    interest_line = [calculate_market_index(x, inflation_rate, gdp_growth) for x in x_vals]
    inflation_line = [calculate_market_index(interest_rate, x, gdp_growth) for x in x_vals]
    gdp_line = [calculate_market_index(interest_rate, inflation_rate, x) for x in np.linspace(-5, 10, 100)]

    ax3.plot(x_vals, interest_line, label="Interest Rate Impact", color='orange')
    ax3.plot(x_vals, inflation_line, label="Inflation Rate Impact", color='green')
    ax3.plot(np.linspace(-5, 10, 100), gdp_line, label="GDP Growth Impact", color='purple')
    ax3.set_ylim(50, 200)
    ax3.set_ylabel("Market Index")
    ax3.set_xlabel("Parameter Value")
    ax3.set_title("Parameter-wise Sensitivity on Market Index")
    ax3.legend()
    st.pyplot(fig3)