import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Market Index Line Graph", layout="centered")

# Title
st.title("📈 Dynamic Market Index Line Graph")
st.write("Adjust the parameters below and click 'Add Point' to track how the Market Index changes over time.")

# Function to calculate market index
def calculate_market_index(interest_rate, inflation_rate, gdp_growth):
    # Define weight factors for impact
    interest_impact = -1.5  # Higher interest rate lowers index
    inflation_impact = -1.2  # Higher inflation lowers index
    gdp_impact = 2.0         # Higher GDP raises index
    
    # Market index formula
    market_index = 100 + (interest_impact * interest_rate) + (inflation_impact * inflation_rate) + (gdp_impact * gdp_growth)
    
    # Clamp market index in range
    return max(50, min(market_index, 200))

# Initialize session state to store data points
if "index_history" not in st.session_state:
    st.session_state.index_history = []
    st.session_state.step_count = 0

# Sliders
col1, col2, col3 = st.columns(3)
with col1:
    interest_rate = st.slider("Interest Rate (%)", 0.0, 15.0, 5.0, 0.1)
with col2:
    inflation_rate = st.slider("Inflation Rate (%)", 0.0, 15.0, 5.0, 0.1)
with col3:
    gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 2.0, 0.1)

# Button to add a new prediction point
if st.button("➕ Add Point"):
    new_index = calculate_market_index(interest_rate, inflation_rate, gdp_growth)
    st.session_state.index_history.append(new_index)
    st.session_state.step_count += 1

# Button to reset the graph
if st.button("🔄 Reset Graph"):
    st.session_state.index_history = []
    st.session_state.step_count = 0

# Plot the line graph
if st.session_state.index_history:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = list(range(1, len(st.session_state.index_history) + 1))
    y_vals = st.session_state.index_history

    ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue', label="Market Index")
    ax.scatter(x_vals[-1], y_vals[-1], color='red', s=100, zorder=5, label="Latest Point")  # Red dot at the latest point
    ax.set_title("Market Index Trend")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("Market Index")
    ax.set_ylim(50, 200)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"### 📍 Latest Predicted Market Index: {y_vals[-1]:.2f}")
else:
    st.info("Use the sliders and click 'Add Point' to start plotting the Market Index trend.")
