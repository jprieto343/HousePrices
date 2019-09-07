# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:30:52 2019

@author: Josue Prieto
"""

import pandas as pd
from tls import column_encoder, clean_dataset

#Loading data

df = pd.read_csv('train.csv')

nullv = df.isnull().sum()
print('Check dtypes and null values')
print(nullv)

def preproc(df):
#   Dropping values that cointain nan's

    df = df.drop(columns=['PoolQC','MiscFeature','Alley','Fence'])
# Dividing values

#   String
    obj_df = df.select_dtypes(include=['object'])

#   Numerical
    num_df = df.select_dtypes(exclude=['object'])

#   Filling nan's
    obj_df.fillna(value='NA')


    columns = obj_df.columns.tolist()

#   Encoding 
    obj_df = column_encoder(obj_df,columns)


    print(obj_df.isnull().sum())

#   Merge the two dataframes
    full_df = pd.concat([obj_df,num_df],axis=1)

    return full_df

full_df = clean_dataset(preproc(df))

# Split into X and Y

X = full_df.drop(columns=['SalePrice'])

y = full_df['SalePrice']

#   Regression time

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor

regresor = DecisionTreeRegressor(max_depth=15)

rfr = RandomForestRegressor()

rfr.fit(X,y)
regresor.fit(X,y)

#   Scoring the regresor


test = pd.read_csv('test.csv')

test = clean_dataset(preproc(test))

X_test = full_df.drop(columns=['SalePrice'])

y_test = full_df['SalePrice']
y_test = y_test.reset_index(drop=True)


y_pred_rfr = rfr.predict(X_test)
y_pred_rfr = pd.Series(y_pred_rfr)
y_pred_rfr = y_pred_rfr.reset_index(drop=True)

y_pred = regresor.predict(X_test)
y_pred = pd.Series(y_pred)
y_pred = y_pred.reset_index(drop=True)
from sklearn import metrics


#   Computing the error

print("Decision Tree RMSE:", math.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Decision Tree MSLE: ',metrics.mean_squared_log_error(y_test, y_pred)) 

print("Random Forest RMSE:", math.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)))

print('Random Forest MSLE: ',metrics.mean_squared_log_error(y_test, y_pred_rfr)) 

plt.figure(figsize=(14,9))
plt.plot(y_test[10:60], label='True Value')
plt.plot(y_pred[10:60],label='Predicted Value Decision Tree')
plt.plot(y_pred[10:60],label='Predicted Value Random Forest')
plt.legend()
plt.grid()
plt.plot()


data = {'prediction': y_pred, 'value': y_test}
data = pd.DataFrame(data=data)
data.to_csv('results.csv',index=False)