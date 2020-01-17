# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:07:38 2020

@author: sriqu
"""
#my responsibilities are Fence,MiscFeature,SaleType,SaleCondition
import pandas as pd

def feature_extract(data) ->alldummies:
    #Fence data
    fenceDummies=pd.get_dummies(data['Fence'])

    #MiscFeature Data
    miscDummies=pd.get_dummies(data['MiscFeature'])


    #Sale Type
    #print(data['SaleType'].isnull().sum()) 
    #There are no missing values, no need to change anything
    saleTypeDummies=pd.get_dummies(data['SaleType'])

    #Sale Condition
    #print(data['SaleCondition'].isnull().sum())
    #No missing values where found
    saleConditionDummies=pd.get_dummies(data['SaleCondition'])
    
    alldummies=pd.concat([fenceDummies,miscDummies,saleTypeDummies,saleConditionDummies])
    
    return alldummies 