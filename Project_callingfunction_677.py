#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Project_677_datacleaning import datacleaning
df=datacleaning()
from Linear_Regression import LinearRegression
linear_model,df = LinearRegression(df)
from Project_677_RandomForest import RandomForest
rf_model=RandomForest(df)
from Project_677_2023data import data_preprocessor
data_23=data_preprocessor()
from Project_677_2023 import final_predictions
final_predictions(df,data_23,linear_model,rf_model)


# In[ ]:




