import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.header("Solar Energy Dashboard")
    # Paste your solar energy dashboard code here
# Constants for installation cost calculation
PANEL_COST_PER_WATT = {"Residential": 50, "Commercial": 45}  # ₹ per watt
INVERTER_COST = 40_000  # ₹ (fixed cost for inverter)
BATTERY_COST_PER_KWH = 15_000  # ₹ per kWh (optional)
STRUCTURE_COST = 10_000  # ₹ (fixed cost for mounting structure)
WIRING_ACCESSORIES_COST = 12_000  # ₹ (fixed cost for wiring and accessories)
INSTALLATION_LABOR_COST = 25_000  # ₹ (fixed cost for labor)

# Constants for panel types and electrical characteristics
RESIDENTIAL_PANEL_AREA = 1.6  # m² (for 60-cell panels)
COMMERCIAL_PANEL_AREA = 2.0  # m² (for 72-cell panels)
VOLTAGE_PER_PANEL = 30  # Typical voltage for a panel in series connection
CURRENT_PER_PANEL = 8  # Typical current for a panel in parallel connection

# Streamlit Title
st.title("Solar Dashboard: Energy Prediction, System Sizing, & Cost Calculator")

# Ensure calculated_system_size is always initialized globally before use
calculated_system_size = 1.0  # Default minimum value to avoid NameError

# User Inputs for System Sizing
energy_consumption = st.number_input("Enter daily energy consumption (kWh):", min_value=1.0, value=30.0)
sunlight_hours = st.slider("Enter peak sunlight hours per day:", min_value=3, max_value=8, value=5)
efficiency_factor = st.slider("Enter system efficiency factor (%):", min_value=70, max_value=100, value=90) / 100

# Perform Calculation ONLY if values are valid
if energy_consumption > 0 and sunlight_hours > 0 and efficiency_factor > 0:
    calculated_system_size = energy_consumption / (sunlight_hours * efficiency_factor)

st.write(f"Recommended System Size: {calculated_system_size:.2f} kW")

# File Upload Section for Dashboard
st.header("Solar Energy Prediction Dashboard")
uploaded_file = st.file_uploader("Upload your solar irradiance dataset (Excel file):", type=["xlsx"])

if uploaded_file:
    # Load the dataset
    data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    data.columns = data.columns.str.strip().str.lower()  # Normalize column names to lowercase
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Convert Excel date to datetime

    # User Inputs for Panel Type and Total Area
    panel_type = st.radio("Select Panel Type:", options=["Residential (60-cell)", "Commercial (72-cell)"])
    total_area = st.number_input("Enter Total Panel Area (in m²):", min_value=1.0, value=10.0)

    # Set panel area based on selection
    if panel_type == "Residential (60-cell)":
        panel_area_per_unit = RESIDENTIAL_PANEL_AREA
    else:
        panel_area_per_unit = COMMERCIAL_PANEL_AREA

    # Calculate number of panels
    num_panels = int(total_area / panel_area_per_unit)
    st.write(f"Number of {panel_type} panels connected: {num_panels}")

    # Calculate parallel and series connections
    num_parallel = st.slider("Enter the number of panels in parallel connection:", min_value=1, max_value=num_panels, value=1)
    num_series = num_panels // num_parallel
    st.write(f"Parallel Connections: {num_parallel}")
    st.write(f"Series Connections: {num_series}")

    # Electrical characteristics
    total_voltage = VOLTAGE_PER_PANEL * num_series
    total_current = CURRENT_PER_PANEL * num_parallel
    st.write(f"Total Voltage (V): {total_voltage}")
    st.write(f"Total Current (A): {total_current}")

    # User Input for Efficiency
    panel_efficiency = st.slider("Select Solar Panel Efficiency (as a percentage):", min_value=5, max_value=25, value=18) / 100

    # Predict Solar Irradiance for 2025–2030
    future_dates = pd.date_range(start='2025-01-01', end='2030-12-31', freq='D')
    future_data = pd.DataFrame({
        'date': future_dates,
        'solar irradiance': np.random.choice(data['solar irradiance'], size=len(future_dates)) * 100  # Simulated irradiance proxy
    })

    # Calculate Solar Energy Output
    future_data['solar energy (kWh/day)'] = (
        future_data['solar irradiance'] * panel_efficiency * total_area
    )

    # Display Predictions
    st.write("Predicted Solar Energy (kWh/day) for 2025–2030:")
    st.dataframe(future_data[['date', 'solar energy (kWh/day)']])

    # Visualization
    st.write("Solar Energy Prediction Plot:")
    plt.figure(figsize=(10, 6))
    plt.plot(future_data['date'], future_data['solar energy (kWh/day)'], label="Predicted Energy")
    plt.xlabel("Date")
    plt.ylabel("Solar Energy (kWh/day)")
    plt.title("Predicted Solar Energy (2025–2030)")
    plt.legend()
    st.pyplot(plt)

# Divider for Calculator
st.markdown("---")

# Cost Calculator Section
st.header("Solar Installation Cost Calculator")
panel_type_calc = st.radio("Select Panel Type for Cost Calculation:", options=["Residential", "Commercial"])
include_battery = st.checkbox("Include Battery Storage?")
battery_capacity_kwh = (
    st.number_input("Enter Battery Capacity (kWh):", min_value=1.0, value=5.0)
    if include_battery
    else 0
)

# Cost Calculation
panel_cost = PANEL_COST_PER_WATT[panel_type_calc] * calculated_system_size * 1000
inverter_cost = INVERTER_COST
battery_cost = battery_capacity_kwh * BATTERY_COST_PER_KWH
total_cost = (
    panel_cost
    + inverter_cost
    + battery_cost
    + STRUCTURE_COST
    + WIRING_ACCESSORIES_COST
    + INSTALLATION_LABOR_COST
)

# Cost Breakdown Display
st.write("### Cost Breakdown:")
st.write(f"**Solar Panels:** ₹{panel_cost:,.2f}")
st.write(f"**Inverter:** ₹{inverter_cost:,.2f}")
st.write(f"**Battery (if included):** ₹{battery_cost:,.2f}")
st.write(f"**Mounting Structure:** ₹{STRUCTURE_COST:,.2f}")
st.write(f"**Wiring & Accessories:** ₹{WIRING_ACCESSORIES_COST:,.2f}")
st.write(f"**Installation & Labor:** ₹{INSTALLATION_LABOR_COST:,.2f}")
st.write(f"#### **Total Estimated Cost:** ₹{total_cost:,.2f}")

# Subsidy Calculation (Optional)
subsidy_percent = st.slider("Select Government Subsidy (%):", min_value=0, max_value=40, value=20)
subsidy_amount = (subsidy_percent / 100) * total_cost
final_cost = total_cost - subsidy_amount

# Display Final Cost
st.write("### Final Cost After Subsidy:")
st.write(f"**Government Subsidy:** ₹{subsidy_amount:,.2f}")
st.write(f"**Final Cost:** ₹{final_cost:,.2f}")
