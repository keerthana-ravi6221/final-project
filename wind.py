import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.header("Wind Energy Dashboard")
    # Paste your wind energy dashboard code here
# Streamlit Title
st.title("Wind Energy Prediction Dashboard")

# File Upload Section
uploaded_file = st.file_uploader("Upload your wind speed dataset (Excel file):", type=["xlsx"])

if uploaded_file:
    # Load the dataset
    data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    data['DATE'] = pd.to_datetime(data['DATE'])

    # User Inputs for Turbine Parameters
    num_turbines = st.number_input("Enter the number of turbines:", min_value=1, value=10)
    swept_area = st.number_input("Enter the swept area of each turbine (in m²):", min_value=1.0, value=50.0)
    efficiency = st.slider("Enter turbine efficiency (as a percentage):", min_value=1, max_value=100, value=40) / 100
    air_density = 1.225  # kg/m³ (constant air density)

    # Predicting Wind Energy for 2025-2030
    future_dates = pd.date_range(start='2025-01-01', end='2030-12-31', freq='D')
    future_data = pd.DataFrame({
        'DATE': future_dates,
        'Max Wind Speed (mps)': np.random.choice(data['Max Wind Speed (mps)'], size=len(future_dates))
    })

    # Wind Energy Calculation
    future_data['Wind Power (W)'] = 0.5 * air_density * swept_area * (future_data['Max Wind Speed (mps)'] ** 3) * efficiency
    future_data['Energy per Turbine (kWh/day)'] = future_data['Wind Power (W)'] * 24 / 1000  # Convert to kWh/day
    future_data['Total Energy (kWh/day)'] = future_data['Energy per Turbine (kWh/day)'] * num_turbines

    # Display Predictions
    st.write("Predicted Total Wind Energy Harnessed (kWh/day) for 2025–2030:")
    st.dataframe(future_data[['DATE', 'Total Energy (kWh/day)']])

    # Visualization
    st.write("Wind Energy Prediction Plot:")
    plt.figure(figsize=(10, 6))
    plt.plot(future_data['DATE'], future_data['Total Energy (kWh/day)'], label="Total Energy Harnessed")
    plt.xlabel("Date")
    plt.ylabel("Total Energy (kWh/day)")
    plt.title("Predicted Wind Energy (2025–2030)")
    plt.legend()
    st.pyplot(plt)
