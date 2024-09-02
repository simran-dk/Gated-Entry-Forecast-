#!/usr/bin/env python
# coding: utf-8

# In[7]:


def datacleaning():
    import pandas as pd

    #Read CSV file into a dataframe
    df = pd.read_csv('/Users/apple/Downloads/GSE_by_year/GSE_2020.csv')
    
    #Filter the DataFrame for the 'Blue Line' and create a copy
    filtered_df = df[df['route_or_line'] == 'Blue Line'].copy()
    
    #Convert 'service_date' column to datetime using .loc
    filtered_df.loc[:, 'service_date'] = pd.to_datetime(filtered_df.loc[:, 'service_date'])
    
    #Group by 'service_date' and 'station_name' and then sum 'gated_entries'
    combined_df_2020 = filtered_df.groupby(['service_date', 'station_name'], as_index=False)['gated_entries'].sum()
    
    #Repeat the process for 2021 and 2022
    
    #2021
    df = pd.read_csv('/Users/apple/Downloads/GSE_by_year/GSE_2021.csv')
    filtered_df = df[df['route_or_line'] == 'Blue Line'].copy()
    filtered_df.loc[:, 'service_date'] = pd.to_datetime(filtered_df.loc[:, 'service_date'])
    combined_df_2021 = filtered_df.groupby(['service_date', 'station_name'], as_index=False)['gated_entries'].sum()
    
    #2022
    df = pd.read_csv('/Users/apple/Downloads/GSE_by_year/GSE_2022.csv')
    filtered_df = df[df['route_or_line'] == 'Blue Line'].copy()
    filtered_df.loc[:, 'service_date'] = pd.to_datetime(filtered_df.loc[:, 'service_date'])
    combined_df_2022 = filtered_df.groupby(['service_date', 'station_name'], as_index=False)['gated_entries'].sum()
    
    #combine the three df for 2020, 2021 and 2022 to give the final df
    combined_df = pd.concat([combined_df_2020, combined_df_2021, combined_df_2022], axis=0)  # Stack vertically (row-wise)
    
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("Data cleaning did not produce a valid DataFrame.")

    return combined_df


# In[ ]:




