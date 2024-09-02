#!/usr/bin/env python
# coding: utf-8

# In[1]:


def LinearRegression(df):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    def preprocess_data(df):
        df['service_date'] = pd.to_datetime(df['service_date']) #convert the service_date to the date time format
        df['year'] = df['service_date'].dt.year #extract the year from the service date 
        df['day_of_year'] = df['service_date'].dt.dayofyear #extract the day from the service date

        #encode the categorical variable to numbers so the model can extract information from it
        label_encoder = LabelEncoder()
        df['station_name_encoded'] = label_encoder.fit_transform(df['station_name']) 

        return df
    
    def train_linear_regression_model(X, y,df):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) #Scale the feature data
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split into training and test dataset
    
        model = LinearRegression() 
        model.fit(X_train, y_train) #fit the linear model to the data
    
        predictions = model.predict(X_test) #predict the values for the test set
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error for Linear Regression Model: {mse}') #Display mean squared error for the model
    
        return model,df #return the trained model and dataframe
    
    #Preprocess the data
    df = preprocess_data(df)
    
    #Prepare the features and output variable
    X = df[['year', 'day_of_year', 'station_name_encoded']]
    y = df['gated_entries']
    
    #Train the linear regression model
    trained_model,df = train_linear_regression_model(X, y,df)
    return trained_model, df
    


# In[ ]:




