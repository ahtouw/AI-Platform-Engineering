# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:58:50 2020

@author: dsene
"""

import pandas as pd
import numpy as np

def ordinal_replace(
    data_sr: pd.DataFrame, 
    dic={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}, 
    filler:int = 0
) -> pd.Series:
    '''
    Parameters:
    data_sr: Pandas series from which we are want to apply the rating
    dic: The dictionary that contains the key of the rating we want to map to the data series
        Ex - 5
        Gd - 4
        TA - 3
        Fa - 2
        Po - 1
        Na - 0
    filler: Replaces all the nan/null values to 0 by default.
    Return:
    Pandas Series of int64 type
    '''
    data_sr.fillna('Na', inplace=True)
    features = ['BsmtCond','BsmtQual','FireplaceQu','GarageQual','GarageCond','PoolQC','ExterQual','KitchenQual','HeatingQC']
    data = data_sr(features)
    # return data_sr.replace(dic).apply(int)
    for col in data:
        data[col].replace(dic, inplace=True)
        
    data_sr.fillna(filler, inplace=True)
    return data_sr[1]