#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:22 2020

@author: staxx

Brandon Jackson
"""

# Function we worked on that was not used
def feature_extract(df):
    '''
    

    Take in dataframe object and check for NaN values and handle
    ----------
    df_train : dataframe
        training dataset from the Ames,Iowa housing file.

    Returns concatenated dataframe 
    -------
    dataframe object of extracted data and transformed variable.

    Looked for missing values and outliers and did not find any.
    
    
    '''
    
# Handle NaN values
    for col in df:
        if col.isnull():
# Condition1 and Condition2 unique values
# Neighborhood Nominal
# LandSlope Ordinal
    
# create list of my respective values for evaluating 
    predictor_columns = ['Neighborhood','Condition1','Condition2']

# create dummy list of relevant values for evaluation
    df_housing = pd.get_dummies(df[predictor_columns])

# based off EDA LandSLope can be changed to numerical based of th three types
    df_housing.LandSlope.replace({'Sev':3, 'Mod':2, 'Gtl':1}, inplace=True)
    fill = 1 
    return pd.concat(df_housing, df.LandSlope)


