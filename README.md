# Gated Entry Prediction for MBTA Blue Line

## Overview
This project aims to predict the number of entries to a particular gated station on the Blue Line of the MBTA train system for the year 2023. The predictions are based on historical data from 2020 to 2022. We employed three different models to achieve this goal: Linear Regression, Random Forest, and ARIMA.

## Problem Statement
Predict the number of entries on a given day to a particular gated station of the Blue Line of the MBTA train system in 2023 based on historical data from 2020-2022.

### Data
	•	Source: Historical entry data for the Blue Line of the MBTA train system.
	•	Preprocessing: Filtered data to include only the Blue Line, formatted the data, and aggregated entries by station and date.
## Models Used
### 1. Linear Regression
	•	Description: Used to predict entries based on encoded categorical variables (station names).
	•	Performance:
	•	Mean Squared Error (MSE): 3,156,086.25
	•	Conclusion: The Linear Regression model showed poor performance with high MSE.
### 2. Random Forest
	•	Description: An ensemble learning method using multiple decision trees to improve prediction accuracy.
	•	Performance:
	•	Mean Squared Error (MSE): 368,337.24
	•	Conclusion: While better than Linear Regression, Random Forest still had significant error.
### 3. ARIMA (Autoregressive Integrated Moving Average)
	•	Description: A time series forecasting method that models the data based on its past values, differencing to achieve stationarity, and smoothing fluctuations using moving averages.
	•	Implementation:
	•	Used ARIMA from the statsmodels library.
	•	Automated parameter tuning with auto_arima.
	•	Performance:
	•	Mean Squared Error (MSE): Significantly lower than other models.
	•	Conclusion: ARIMA demonstrated the best performance in terms of MSE and MAE but with higher complexity and running time. It is crucial to ensure data stationarity for ARIMA to perform well.

## Choosing the Appropriate Model
Based on the MSE values and actual vs. predicted value plots:
	•	ARIMA provided the lowest MSE and MAE, indicating the best predictive performance.
	•	Linear Regression and Random Forest showed higher error rates, with ARIMA outperforming both.

## Conclusion
The ARIMA model is the most effective for predicting gated entries based on its lower MSE and MAE. However, its computational complexity and need for stationary data must be considered. Future work could focus on optimizing ARIMA's performance further or exploring other time series forecasting methods.
## Usage
	1.	Clone the Repository: bashCopy code  git clone https://github.com/yourusername/gated-entry-prediction.git	  
	
 	2.	Navigate to the Project Directory: bashCopy code  cd gated-entry-prediction
		  
	3.	Install Dependencies: bashCopy code  pip install -r requirements.txt
		  
	4.	Run the Models:
 
  		• 	Data Preprocessing: python Project_677_datacleaning.py and Project_677_2023data.py
   
		•	Linear Regression: python Project_677_Linear_Regression.py

		•	Random Forest: python Project_677_RandomForest.py
 
		•	ARIMA:python Project_677_ensemble.py
 
 		•	Calling Function:python Project_callingfunction_677.py

## Requirements
	•	Python 3.x
	•	numpy
	•	pandas
	•	scikit-learn
	•	statsmodels

## Contributing
Feel free to fork the repository, make improvements, and submit pull requests. For any issues or suggestions, please open an issue or contact the project maintainer.
