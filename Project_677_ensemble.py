#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tkinter as tk
from tkinter import ttk
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def final_predictions(df,data_23, Linear_model, rf_model):
    #ARIMA model
    #Display all available station names
    print("Available station names:")
    print(df['station_name'].unique())
    
    print('For ARIMA model:')
    #Accept the station name as input from user 
    station_name = input("Enter the station name: ")
    station_data = df[df['station_name'] == station_name].copy()
    combined_data=data_23[data_23['station_name'] == station_name].copy()
    
    if station_data.empty:
        print(f"No data found for the station: {station_name}")
        return

    #Create a time series with 'service_date' as the index
    time_series = station_data.set_index('service_date')['gated_entries']
    
    #Train the ARIMA model on the time series generated above
    arima_model = auto_arima(time_series, suppress_warnings=True, seasonal=False)
    arima_fit = arima_model.fit(time_series)
    
    #Take input for the date from the user
    date_str = input("Enter the date (YYYY-MM-DD) for prediction in 2023 (only uptill 2023-04-30): ")
    date = pd.to_datetime(date_str)
    
    #set the entered date as the end of the forecast period for the ARIMA model
    forecast_start = pd.to_datetime('2022-12-31')
    forecast_periods = (date - forecast_start).days
    arima_predictions = arima_fit.predict(n_periods=forecast_periods)

    #Extract and store the actual values of gated entries
    combined_data_from_date = combined_data[(combined_data['service_date'] > forecast_start) & (combined_data['service_date'] <= date)].copy()
    actual_values = combined_data_from_date['gated_entries']
    
    #Process the data for machine learning models
    data_23=data_23[data_23['station_name'] == station_name].copy()
    combined_data23_from_date = data_23[(data_23['service_date'] > forecast_start) & (data_23['service_date'] <= date)].copy()
    
    combined_data23_from_date['service_date'] = pd.to_datetime(combined_data23_from_date['service_date'])
    combined_data23_from_date['year'] = combined_data23_from_date['service_date'].dt.year
    combined_data23_from_date['day_of_year'] = combined_data23_from_date['service_date'].dt.dayofyear
    label_encoder = LabelEncoder()
    combined_data23_from_date['station_name_encoded'] = label_encoder.fit_transform(combined_data23_from_date['station_name'])
    
    #Extract the features from the dataset and store as X and the true values as Y
    X = combined_data23_from_date[['year', 'day_of_year', 'station_name_encoded']]
    y = combined_data23_from_date['gated_entries']
    
    #Linear Regression model
    #Use X as input for the trained linear model 
    linear_predictions = Linear_model.predict(X)
    linear_mse = mean_squared_error(y, linear_predictions)
    
    #Random Forest Model
    #Use X as input for the trained Random Forest model
    rf_predictions = rf_model.predict(X)
    rf_mse = mean_squared_error(y, rf_predictions)

    
    #Visualize ARIMA predictions
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data_from_date['service_date'], actual_values, label='Actual Values')
    plt.plot(pd.date_range(start=forecast_start, periods=len(arima_predictions), freq='D'), arima_predictions, label='ARIMA Predictions')
    plt.title(f'ARIMA Predictions vs Actual Values for {station_name}')
    plt.xlabel('Date')
    plt.ylabel('Gated Entries')
    plt.legend()
    plt.show()

    #Visualize Linear Model predictions
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data23_from_date['service_date'], y, label='Actual Values')
    plt.plot(combined_data23_from_date['service_date'], linear_predictions, label='Linear Regression Predictions')
    plt.title(f'Linear Regression Predictions vs Actual Values for {station_name}')
    plt.xlabel('Date')
    plt.ylabel('Gated Entries')
    plt.legend()
    plt.show()

    #Visualize Random Forest predictions
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data23_from_date['service_date'], y, label='Actual Values')
    plt.plot(combined_data23_from_date['service_date'], rf_predictions, label='Random Forest Predictions')
    plt.title(f'Random Forest Predictions vs Actual Values for {station_name}')
    plt.xlabel('Date')
    plt.ylabel('Gated Entries')
    plt.legend()
    plt.show()
    
    #Display Mean Absolute Error
    mae_arima = mean_absolute_error(actual_values, arima_predictions)
    mae_lm = mean_absolute_error(y, linear_predictions)
    mae_rf = mean_absolute_error(y, rf_predictions)
    print(f'The mean absolute error for ARIMA is = {mae_arima}')
    print(f'The mean absolute error for Linear Regression model is = {mae_lm}')
    print(f'The mean absolute error  Random Forest is = {mae_rf}')
    
    #Choosing the model with the least mean absolute error
    min_mae_model = min([(mae_arima, 'ARIMA'), (mae_lm, 'Linear Regression'), (mae_rf, 'Random Forest')], key=lambda x: x[0])
    min_mae, min_mae_model_name = min_mae_model

    #Print the predictions for the model with the least MAE
    if min_mae_model_name == 'ARIMA':
        print(f'ARIMA Predicted value for {date.date()}: {arima_predictions.iloc[-1]}')
    elif min_mae_model_name == 'Linear Regression':
        print(f'Linear Regression Predicted value for {date.date()}: {linear_predictions.iloc[-1]}')
    elif min_mae_model_name == 'Random Forest':
        print(f'Random Forest Predicted value for {date.date()}: {rf_predictions.iloc[-1]}')

