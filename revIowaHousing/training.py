# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:26:08 2020

@author: Corey Paisley
"""
'''
--- This file is the main execution file ---
It covers:
(1)EXTERNAL INPORTATION of csv, libraries, and other dev. functions,
(2)ALL FEATURE ENGINEERING(Dropping col, fillNA, skewAdjust, Dummys)
(3)MODEL EXECUTION on refined data
 
'''
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
import pandas as pd
import numpy as np
from ordinals import ordinals
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from joblib import load, dump
from unique.fence import fence_uniq
from unique.cond_2hot import conditions_2hot
from unique.bsmtfn_type import basement_type
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Read the dataset, get our predicted variable, then remove it from the training set
# Stephen Wight
df_in = pd.read_csv('train.csv')
SalePrice = df_in.SalePrice.apply(lambda x: x**0.5) # John Winegardener suggested taking sqrt
df_in.drop('SalePrice',axis=1,inplace=True)

# Remove columns we decided not to use
df_in.drop(['Id','Utilities','Heating', 'KitchenAbvGr', '3SsnPorch','Exterior2nd','TotalBsmtSF'], axis=1, inplace=True)

## ORDINAL FEATURES
# Handle ordinal columns, then return the dataset as numeric and nominal
df_num, df_obj = ordinals(df_in)

## UNIQUE FEATURES THAT DON'T FIT WELL INTO ANY SINGLE CATEGORY
# Handle unique variables, then drop them
# Stephen Wight wrote the calls to these functions
df_fence = fence_uniq(df_in)
df_num = pd.concat([df_num, df_fence], axis=1)

df_cond = conditions_2hot(df_in)
df_num = pd.concat([df_num, df_cond], axis=1)

df_bsmt = basement_type(df_in)
df_num = pd.concat([df_num, df_bsmt], axis=1)

df_obj.drop(['Fence','BsmtFinType1','BsmtFinType2', 'Condition1','Condition2'], axis=1, inplace=True)

# Fill all nominal variables that have missing data with 'None', as it is safer to assume they
# didn't have it than to assume a value. --Stephen Wight
df_obj.fillna('None', inplace=True)

# Get dummy columns for all of the remaining nominal variables. Note that we are asking
# the model to ignore things it doesn't know, so that it will process unexpected variation
# in the predicting phase. - Stephen Wight

enc1h = OneHotEncoder(sparse=False, handle_unknown='ignore')
df_obj = pd.DataFrame(enc1h.fit_transform(df_obj))
# Save the trained model.
dump(enc1h, 'OneHotEnc.joblib') # Michael Sriqui

## NUMERICAL FEATURES

# Get the medians of the columns, then save them in a list of column-headings
col_medians = df_num.median(axis=0) # Group Decision, Stephen Wight coded
dump(col_medians,'ColumnMedians.joblib')

# Fill those columns with their respective medians
df_num.fillna(col_medians, inplace=True) # Group Decision, Stephen Wight coded

# Adjust for skewed variables, and save a record of which ones we adjust # Group decision, Stephen Wight coded
df_num_skew = df_num.skew(axis=0)
df_num_skew = df_num_skew.loc[df_num_skew > 0.75].index
dump(df_num_skew,'SkewCols.joblib')
df_num[df_num_skew] = df_num[df_num_skew].apply(np.log1p)

## Pull all of the columns together
df_final = pd.concat([df_obj, df_num],axis=1)

### Normalization ### Coded by Stephen Wight

scaler = MinMaxScaler()
np_final = scaler.fit_transform(df_final)
dump(scaler,'normalizer.joblib') # Michael Sriqui

### PCA ### Coded by Will Ah Tou

pcaobj = PCA(n_components=0.95)
np_final = pcaobj.fit_transform(np_final)
dump(pcaobj, 'pcaObject.joblib') # Michael Sriqui

### TRAIN/TEST SPLIT ### Coded by Stephen Wight

X_train, X_test, y_train, y_test = train_test_split(
    np_final,
    SalePrice,
    test_size = 0.2,
)

### ML MODELS ### Coded by Corey Paisley, hyperparameter adjustments made as a group

#See L.Regres Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
lm = LinearRegression(
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    n_jobs=None)

lm.fit(X_train, y_train)
dump(lm, 'OLS.joblib') # Michael Sriqui

#See Ridge Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge 
lr = Ridge(
    alpha=1.0,
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    max_iter=None, 
    tol=0.001,
    solver='auto', 
    random_state=None)

lr.fit(X_train, y_train)
dump(lr, 'Ridge.joblib') # Michael Sriqui

#See SGD Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
sgd = SGDRegressor(
    loss='squared_loss',
    penalty='l2',
    alpha=0.0001, 
    l1_ratio=0.15,
    fit_intercept=True, 
    max_iter=1000,
    tol=0.001, 
    shuffle=True, 
    verbose=0,
    epsilon=0.1, 
    random_state=None,
    learning_rate='invscaling', 
    eta0=0.01,
    power_t=0.25, 
    early_stopping=False,
    validation_fraction=0.1, 
    n_iter_no_change=5,
    warm_start=False, 
    average=False)

sgd.fit(X_train, y_train)
dump(sgd, 'SGD.joblib') # Michael Sriqui

print('OLS\t\t', lm.score(X_test,y_test))
print('Ridge\t\t',lr.score(X_test,y_test))
print('SGD\t\t', sgd.score(X_test,y_test))