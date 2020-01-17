#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:53:36 2020

@author: staxx

Brandon Jackson

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn as sklearn


# import full training set
df_housing = pd.read_csv("/home/staxx/Documents/Iowa_House/train.csv")

# get information about each column
df_housing.info()

# copy of dataframe for just the columns I need
#fObj = pd.DataFrame(df_housing, columns = ['Neighborhood' , 'LandSlope', 'Condition1' , 'Condition2', 'SalePrice'])



# create dataframe of training set predictors train_x
# drop unnecessary columns
df_housing = df_housing.drop(["SaleCondition","SaleType","YrSold","MoSold","MiscVal","MiscFeature","Fence","PoolQC","PoolArea","ScreenPorch","3SsnPorch","EnclosedPorch","OpenPorchSF","WoodDeckSF","PavedDrive","GarageCond","GarageQual","GarageArea","GarageCars","GarageFinish","GarageYrBlt","GarageType","FireplaceQu","Fireplaces","Functional","TotRmsAbvGrd","KitchenQual","KitchenAbvGr","BedroomAbvGr","HalfBath","FullBath","BsmtHalfBath","BsmtFullBath","GrLivArea","LowQualFinSF","2ndFlrSF","1stFlrSF","Electrical","CentralAir","HeatingQC","Heating","TotalBsmtSF","BsmtUnfSF","BsmtFinSF2","BsmtFinType2","BsmtFinSF1","BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual","Foundation","ExterCond","ExterQual","MasVnrArea","MasVnrType","Exterior2nd","Exterior1st","RoofMatl","RoofStyle","YearRemodAdd","YearBuilt","OverallCond","OverallQual","HouseStyle","BldgType","LotConfig","Utilities","LandContour","LotShape","Alley","Street","LotArea","LotFrontage","MSZoning","MSSubClass"], axis=1, inplace=True)    
df_housing = df_housing.drop(["Id"], axis=1, inplace=True)

# set up correlation matrix components
corrMatrix = df_housing.corr()
# pair plot showing the realtionship and skew of variables
sns.pairplot(df_housing)

# Attempt to adjust for postive skew of LandSlope
helpful_log = np.log(df_housing.LandSlope)
helpful_log.describe()

# display the different values with my variables
# LandSlope is 
df_housing['LandSlope'].value_counts()
# Neighborhood is

df_housing['Neighborhood'].value_counts()
# Condition1 is 
df_housing['Condition1'].value_counts()
# Condition2 is
df_housing['Condition2'].value_counts()

# check to see which values are null if any

# df_housing['Condition2'].isnull().any() etc. for other conditions
#check to see which columns are numerical and categorical
numeric = [col for col in df_housing.columns if df_housing.dtypes[col] != 'object']
categorical = [col for col in df_housing.columns if df_housing.dtypes[col] == 'object']
print(numeric)
print(categorical)

# 25 neighborhoods but want to check distribution of each variable
df_housing['Neighborhood'].hist(bins=30)

# house SalePrice by Neighborhood and other conditions via boxplot

df_housing.boxplot(column='SalePrice', by = 'Neighborhood')
df_housing.boxplot(column='SalePrice', by = 'LandSlope')
df_housing.boxplot(column='SalePrice', by = 'Condition1')
df_housing.boxplot(column='SalePrice', by = 'Condition2')



