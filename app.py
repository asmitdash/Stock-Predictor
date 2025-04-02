import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def calculate_market_index(interest_rate, inflation_rate, gdp_growth):
    # Define weight factors for impact on market index
    interest_impact = -1.5  # Negative impact when interest rate increases
    inflation_impact = -1.2  # Negative impact when inflation increases
    gdp_impact = 2.0  # Positive impact when GDP growth increases
    
    # Calculate market index
    market_index = 100 + (interest_impact * interest_rate) + (inflation_impact * inflation_rate) + (gdp_impact * gdp_growth)
    return max(50, min(market_index, 200))  # Keeping market index in a reasonable range

st.title("Dynamic Market Index Prediction")
st.write("Adjust the parameters below to see how they affect the Market Index.")

# Create sliders for input
interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
inflation_rate = st.slider("Inflation Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
gdp_growth = st.slider("GDP Growth Rate (%)", min_value=-5.0, max_value=10.0, value=2.0, step=0.1)

if st.button("Generate Graph"):
    # Calculate market index
    market_index = calculate_market_index(interest_rate, inflation_rate, gdp_growth)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Market Index"], [market_index], color='blue')
    ax.set_ylim(50, 200)
    ax.set_ylabel("Market Index Value")
    ax.set_title("Market Index Based on Economic Indicators")
    
    # Display the graph and the market index value
    st.pyplot(fig)
    st.write(f"### Predicted Market Index: {market_index:.2f}")






#https://stock-graph.streamlit.app/

