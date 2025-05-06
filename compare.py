import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import timedelta
from io import BytesIO

def show():
    st.header("Comparison Dashboard")
    # Paste your existing comparison logic here

# Load data
def load_data(file_path):
    """
    Loads data from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    df = pd.read_excel(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df


# General forecasting function for any model
def forecast_model(df, model_name='ANN'):
    """
    Forecasts power consumption using the specified model.

    Args:
        df (pd.DataFrame): Input data.
        model_name (str): Name of the model ('ANN', 'RF', 'SVM').

    Returns:
        tuple: (forecast_df, mse, r2, mae, predicted_df)
            forecast_df (pd.DataFrame): DataFrame with forecasted values.
            mse (float): Mean Squared Error.
            r2 (float): R-squared.
            mae (float): Mean Absolute Error.
            predicted_df (pd.DataFrame): DataFrame with predictions vs actuals.
    """
    df['Year'] = df['DATE'].dt.year
    df['DayOfYear'] = df['DATE'].dt.dayofyear

    X = df[['Year', 'DayOfYear']].values
    y = df['Power_Consumption(MU)'].values.reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    if model_name == 'ANN':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    elif model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'SVM':
        model = SVR(C=100, gamma=0.1, epsilon=0.1)
    else:
        raise ValueError("Invalid model name")

    model.fit(X_train, y_train.ravel())

    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_inv = y_scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv, y_pred)
    r2 = r2_score(y_test_inv, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred)

    last_date = df['DATE'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 5 * 365 + 1)]
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Year': [d.year for d in future_dates],
        'DayOfYear': [d.timetuple().tm_yday for d in future_dates]
    })

    future_scaled = x_scaler.transform(future_df[['Year', 'DayOfYear']].values)
    future_predictions_scaled = model.predict(future_scaled)
    future_predictions = y_scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Power_Consumption(MU)': future_predictions.flatten(),
        'Model': model_name
    })

    predicted_df = pd.DataFrame({
        'Date': pd.to_datetime(dict(
            year=(X_test[:, 0] * (x_scaler.data_max_[0] - x_scaler.data_min_[0]) + x_scaler.data_min_[0]).astype(int),
            month=1, day=1)) +
                pd.to_timedelta(((X_test[:, 1] * (x_scaler.data_max_[1] - x_scaler.data_min_[1]) + x_scaler.data_min_[
                    1]) - 1).astype(int), unit='D'),
        'Actual': y_test_inv.flatten(),
        'Predicted': y_pred.flatten(),
        'Model': model_name
    })
    return forecast_df, mse, r2, mae, predicted_df


# Streamlit dashboard
st.title("🔌 Power Consumption Forecasting Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])

if uploaded_file is not None:
    # Read the file
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Ensure 'DATE' column exists
    if 'DATE' not in df.columns:
        st.error("Error: The 'DATE' column is missing in the uploaded file.")
        st.stop()

    df['DATE'] = pd.to_datetime(df['DATE'])  # convert 'DATE' column to datetime

    # Ensure 'Power_Consumption(MU)' column exists
    if 'Power_Consumption(MU)' not in df.columns:
        st.error("Error: The 'Power_Consumption(MU)' column is missing in the uploaded file.")
        st.stop()

    model_choice = st.sidebar.selectbox("Choose Model", ['ANN', 'RF', 'SVM'])

    if st.sidebar.button("Run Forecast"):
        forecast_df, mse, r2, mae, predicted_df = forecast_model(df, model_choice)

        st.subheader(f"Forecasting Results - {model_choice}")
        st.markdown(f"MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")

        st.line_chart(data=forecast_df.set_index('Date')['Power_Consumption(MU)'])

        with st.expander("📋 Predicted vs Actual Data"):
            st.dataframe(predicted_df.head(50))

        with st.expander("📈 Forecasted Future Data"):
            st.dataframe(forecast_df.head(50))

        # Error distribution plot
        st.subheader("📊 Error Distribution")
        predicted_df['Error'] = predicted_df['Actual'] - predicted_df['Predicted']
        fig_error, ax_error = plt.subplots()
        sns.histplot(predicted_df['Error'], bins=30, kde=True, ax=ax_error)
        ax_error.set_title(f"Error Distribution for {model_choice}")
        st.pyplot(fig_error)

        # Daily changes plot
        st.subheader("📉 Daily Power Consumption Changes")
        df_sorted = df.sort_values('DATE')
        df_sorted['Daily Change'] = df_sorted['Power_Consumption(MU)'].diff()
        fig_daily, ax_daily = plt.subplots()
        ax_daily.plot(df_sorted['DATE'], df_sorted['Daily Change'])
        ax_daily.set_title("Daily Change in Power Consumption")
        ax_daily.set_xlabel("Date")
        ax_daily.set_ylabel("Change (MU)")
        st.pyplot(fig_daily)

        # Joint forecast
        st.subheader("📉 Joint Model Comparison (5-Year Forecast)")
        ann_df, ann_mse, ann_r2, ann_mae, _ = forecast_model(df, 'ANN')
        rf_df, rf_mse, rf_r2, rf_mae, _ = forecast_model(df, 'RF')
        svm_df, svm_mse, svm_r2, svm_mae, _ = forecast_model(df, 'SVM')

        joint_df = pd.concat([ann_df.assign(Model='ANN'), rf_df.assign(Model='RF'), svm_df.assign(Model='SVM')])
        fig_joint, ax_joint = plt.subplots(figsize=(14, 7))
        for label, data in joint_df.groupby('Model'):
            ax_joint.plot(data['Date'], data['Power_Consumption(MU)'], label=label)
        ax_joint.legend()
        ax_joint.set_title("Comparison of Forecasts Over 5 Years")
        ax_joint.set_xlabel("Date")
        ax_joint.set_ylabel("Power Consumption (MU)")
        ax_joint.grid(True)
        st.pyplot(fig_joint)

        # Model comparison metrics table
        st.subheader("📊 Model Comparison Metrics")
        metrics_df = pd.DataFrame({
            'Model': ['ANN', 'RF', 'SVM'],
            'MSE': [ann_mse, rf_mse, svm_mse],
            'R²': [ann_r2, rf_r2, svm_r2],
            'MAE': [ann_mae, rf_mae, svm_mae]
        })
        st.dataframe(metrics_df)

        # Provide Excel file download links
        st.subheader("📥 Download Forecast and Prediction Data")
        for model in ['ANN', 'RF', 'SVM']:
            # Create in-memory Excel file
            excel_buffer_pred = BytesIO()
            predicted_df[predicted_df['Model'] == model].to_excel(excel_buffer_pred, index=False)
            excel_buffer_pred.seek(0)  # Reset buffer to the beginning

            excel_buffer_forecast = BytesIO()
            forecast_df[forecast_df['Model'] == model].to_excel(excel_buffer_forecast, index=False)
            excel_buffer_forecast.seek(0)

            st.download_button(
                label=f"Download {model} Actual vs Predicted",
                data=excel_buffer_pred,
                file_name=f"{model}_predicted_vs_actual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.download_button(
                label=f"Download {model} 5-Year Forecast",
                data=excel_buffer_forecast,
                file_name=f"{model}_5yr_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.warning("Please upload an Excel file to proceed.")