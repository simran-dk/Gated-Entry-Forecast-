#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def RandomForest(df):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import LabelEncoder
    
    #Extract features anf target from the dataframe
    X = df[['year', 'day_of_year', 'station_name_encoded']]
    y = df['gated_entries']

    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=32)
    #train random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=32)
    rf_model.fit(X_train, y_train)
    #predict using trained random forest model
    predictions=rf_model.predict(X_test)
    #Calculate and display the mean squared error for the random forest model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error for Random Forest: {mse}')
    return rf_model

