
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
    return max(50, min(market_index, 200))  # Clamp value between 50 and 200

# --- Streamlit UI ---
st.set_page_config(page_title="Market Index Simulation", layout="centered")
st.title("Market Index Simulator")
st.markdown("Use the sliders to simulate how **Interest Rate**, **Inflation**, and **GDP Growth** affect the Market Index.")

# --- Sliders ---
interest_rate = st.slider("Interest Rate (%)", 0.0, 15.0, 5.0, 0.1)
inflation_rate = st.slider("Inflation Rate (%)", 0.0, 15.0, 5.0, 0.1)
gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 2.0, 0.1)

# --- Button to generate updated graphs ---
if st.button("Generate Graph"):
    market_index = calculate_market_index(interest_rate, inflation_rate, gdp_growth)
    st.subheader(f"Predicted Market Index: {market_index:.2f}")

    # --- Graph 1: Scatter plot with red dot ---
    np.random.seed(42)
    actual_values = np.random.uniform(100, 200, 100)
    noise = np.random.normal(0, 5, size=100)
    predicted_values = actual_values + noise

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.scatter(actual_values, predicted_values, label="Sample Predictions", alpha=0.6)
    ax1.plot([100, 200], [100, 200], linestyle='--', color='blue', label='Perfect Fit')
    ax1.scatter(market_index, market_index, color='red', label="Your Prediction", s=100)
    ax1.set_xlabel("Actual Stock Price")
    ax1.set_ylabel("Predicted Stock Price")
    ax1.set_title("Prediction Comparison")
    ax1.legend()
    st.pyplot(fig1)

    # --- Graph 2: Parameter Sensitivity Line Graph ---
    x_vals = np.linspace(0, 15, 100)
    gdp_vals = np.linspace(-5, 10, 100)

    interest_line = [calculate_market_index(x, inflation_rate, gdp_growth) for x in x_vals]
    inflation_line = [calculate_market_index(interest_rate, x, gdp_growth) for x in x_vals]
    gdp_line = [calculate_market_index(interest_rate, inflation_rate, x) for x in gdp_vals]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_vals, interest_line, label="Interest Rate Impact", color='orange')
    ax2.plot(x_vals, inflation_line, label="Inflation Rate Impact", color='green')
    ax2.plot(gdp_vals, gdp_line, label="GDP Growth Impact", color='purple')
    ax2.set_ylim(50, 200)
    ax2.set_ylabel("Market Index")
    ax2.set_xlabel("Parameter Value")
    ax2.set_title("Impact of Each Parameter on Market Index")
    ax2.legend()
    st.pyplot(fig2)