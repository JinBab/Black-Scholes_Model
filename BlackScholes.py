import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# Black-Scholes Option Pricing Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

st.write("""
# Black-Scholes Option Pricing Model
This app allows you to calculate the Black-Scholes option pricing model parameters and visualize the results.


""")


st.sidebar.header("Black-Scholes Model")

st.sidebar.markdown("---")
st.sidebar.markdown("### Created by:")
st.sidebar.markdown("**Jin Wan Kim**")
st.sidebar.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/jinwan-kim-547281217/)")
st.sidebar.markdown("---")
S = st.sidebar.number_input("Current Asset Price", min_value=1.0, value=100.0, step=0.01)
K = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.0, step=0.01)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.05, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate", min_value=0.01, value=0.04, step=0.01)
vol = st.sidebar.number_input("Volatility", value=0.2, step=0.01)

min_vol = st.sidebar.slider("Minimum Volatility", min_value=0.00, max_value=1.00, value=max(vol-0.1, 0.00), step=0.01)
max_vol = st.sidebar.slider("Maximum Volatility", min_value=0.00, max_value=1.00, value=min(vol+0.1, 1.00), step=0.01)

min_strike = st.sidebar.number_input("Min Strike Price", min_value=0.01, value=max(S-20,0.01), step=0.01)
max_strike = st.sidebar.number_input("Max Strike Price", min_value=0.01, value=S+20, step=0.01)



# Add option type selection
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

# Add visualization type selection
viz_type = st.sidebar.selectbox("Visualization Type", ["Heatmap", "3D Surface"])

# Add creator section at the bottom of sidebar


# Create and display parameters table
st.subheader("Current Black-Scholes Parameters")

parameters_df = pd.DataFrame({
    'Parameter': [
        'Current Asset Price (S)',
        'Strike Price (K)', 
        'Time to Maturity (T)',
        'Risk-Free Rate (r)',
        'Volatility (σ)',
        'Option Type'
    ],
    'Value': [
        f'${S:.2f}',
        f'${K:.2f}',
        f'{T:.2f} years',
        f'{r:.1%}',
        f'{vol:.1%}',
        option_type
    ],
    'Description': [
        'Current price of the underlying asset',
        'Exercise price of the option',
        'Time until option expiration',
        'Risk-free interest rate',
        'Implied volatility of the asset',
        'Call or Put option'
    ]
})


st.table(parameters_df)


# Display current option price
current_price = black_scholes_call(S, K, T, r, vol) if option_type == "Call" else black_scholes_put(S, K, T, r, vol)

st.subheader(f"**Current {option_type} Option Price: ${current_price:.2f}**")
# Generate 2D DataFrame for heatmap
st.subheader(f"Black-Scholes Option Price {viz_type}")

# Create ranges for strike prices and volatilities
strike_range = np.linspace(min_strike, max_strike, 10)
vol_range = np.linspace(min_vol, max_vol, 10)

# Create 2D DataFrame
option_prices = pd.DataFrame(index=strike_range, columns=vol_range)

# Fill the DataFrame with Black-Scholes prices
for strike in strike_range:
    for volatility in vol_range:
        if option_type == "Call":
            price = black_scholes_call(S, strike, T, r, volatility)
        else:
            price = black_scholes_put(S, strike, T, r, volatility)
        option_prices.loc[strike, volatility] = price

# Convert to numeric (in case of any string conversion issues)
option_prices = option_prices.astype(float)

# Create visualization based on selected type
if viz_type == "Heatmap":
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=option_prices.values,
        x=vol_range,
        y=strike_range,
        colorscale='Viridis',
        text=np.round(option_prices.values, 2),  # Display rounded values as text
        texttemplate="%{text}",  # Show the text values
        textfont={"size": 10, "color": "white"},  # White text for better visibility
        hovertemplate='Volatility: %{x:.3f}<br>Strike Price: %{y:.2f}<br>Option Price: %{z:.2f}<extra></extra>',
        colorbar=dict(title=f"{option_type} Option Price"),
        hoverongaps=False
    ))

    fig.update_layout(
        title=f"Black-Scholes {option_type} Option Price Heatmap",
        xaxis_title="Volatility",
        yaxis_title="Strike Price",
        width=800,
        height=600
    )

else:  # 3D Surface
    # Create 3D surface plot
    fig = go.Figure(data=go.Surface(
        z=option_prices.values,
        x=vol_range,
        y=strike_range,
        colorscale='Viridis',
        colorbar=dict(title=f"{option_type} Option Price"),
        hovertemplate='Volatility: %{x:.3f}<br>Strike Price: %{y:.2f}<br>Option Price: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Black-Scholes {option_type} Option Price 3D Surface",
        scene=dict(
            xaxis_title="Volatility",
            yaxis_title="Strike Price", 
            zaxis_title=f"{option_type} Option Price",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )

st.plotly_chart(fig)


# Display the DataFrame
st.subheader("Option Price Matrix")
st.write("Rows: Strike Prices, Columns: Volatilities")
st.dataframe(option_prices.round(2))



# data = yf.download(ticker, start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
# data = data.Close

# st.line_chart(data)

