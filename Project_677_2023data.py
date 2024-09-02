#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def data_preprocessor():
    import pandas as pd
    df = pd.read_csv('/Users/apple/Downloads/GSE_by_year/GSE_2023.csv')
    
    #Filter the DataFrame for the 'Blue Line'
    filtered_df = df[df['route_or_line'] == 'Blue Line'].copy()
    
    #Convert 'service_date' column to datetime format
    filtered_df['service_date'] = pd.to_datetime(filtered_df['service_date'])
    
    #Group by 'service_date' and 'station_name' and then add up the 'gated_entries' for all the time stamps of the same date and station name
    combined_df_2023 = filtered_df.groupby(['service_date', 'station_name'], as_index=False)['gated_entries'].sum()
    return combined_df_2023

