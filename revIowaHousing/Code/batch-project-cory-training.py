# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:26:08 2020

@author: paisl
"""
'''
--- This file is a skeleton template for the final training execution file ---
It covers:
(1)EXTERNAL INPORTATION of csv, libraries, and other dev. functions,
(2)ALL FEATURE ENGINEERING(Dropping col, fillNA, skewAdjust, Dummys)
(3)MODEL EXECUTION on refined data
 
'''
from sklearn import linear_model
import pandas as pd
import numpy as np

def oneHotEncode(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a dataframe
    Performs one-hot-encoding
    returns updated dataframe
    '''
    updatedDF = df
# Code here    
    return updatedDF
  
def dropColumns(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a dataframe
    Drops columns
    returns updated dataframe
    '''
    updatedDF = df
# Code here    
    return updatedDF
    
def skewAdjust(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a dataframe
    adjusts for skew in numeric values
    returns updated dataframe
    
    '''
    updatedDF = df
# Code here    
    return updatedDF
    
def fillNA(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a dataframe
    Fills/handles missing or NA values
    returns updated dataframe
    
    '''
    updatedDF = df
# Code here    
    return updatedDF
  
df = pd.read_csv('train.csv')

df = oneHotEncode(df)
df = dropColumns(df)
df = skewAdjust(df)
df = fillNA(df)

'''Models defined here
'''
#See L.Regres Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
lm = linear_model.LinearRegression(fit_intercept=True, 
                                   normalize=False, 
                                   copy_X=True, 
                                   n_jobs=None)

#See Ridge Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge 
lr = linear_model.Ridge(alpha=1.0,
                        fit_intercept=True,
                        normalize=False,copy_X=True,
                        max_iter=None, tol=0.001,
                        solver='auto', random_state=None)

#See SGD Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='l2',
                                alpha=0.0001, l1_ratio=0.15,
                                fit_intercept=True, max_iter=1000,
                                tol=0.001, shuffle=True, verbose=0,
                                epsilon=0.1, random_state=None,
                                learning_rate='invscaling', eta0=0.01,
                                power_t=0.25, early_stopping=False,
                                validation_fraction=0.1, n_iter_no_change=5,
                                warm_start=False, average=False)



