# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:22:21 2020

@author: admin_guest
"""


from mysite import app

if __name__ == '__main__':
        app.run()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_house_price = pd.read_csv('house_price.csv')



Out[5]:
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')

df_house_price has 81 columns (79 features + id and target SalePrice) and 1460 entries (number of rows or house sales)
In [7]:


numerical_var = df_house_price.dtypes[df_house_price.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_var))
​
categorical_var = df_house_price.dtypes[df_house_price.dtypes == "object"]
print("Number of Categorical features: ", len(categorical_var))
#filter numeric column only 
data_num = df_house_price[numerical_var]

#calculating correlation among numeric variable 
corr_matrix = data_num.corr() 

#filter correlation values above 0.5
filter_corr = corr_matrix[corr_matrix > 0.5]

#plot correlation matrix
plt.figure(figsize=(20,12))
sns.heatmap(filter_corr,
            cmap='coolwarm',
            annot=True);
            #scatter plot OverallQual/saleprice
data = pd.concat([df_house_price["SalePrice"], df_house_price["OverallQual"]],axis=1)
data.plot.scatter(x="OverallQual", y="SalePrice", ylim=(0,800000));
#missing data
total_missing_value = df_house_price.isnull().sum().sort_values(ascending=False)
percent_of_missign_value = (df_house_price.isnull().sum()/df_house_price.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_value, percent_of_missign_value], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_house_price["FireplaceQu"] = df_house_price["FireplaceQu"].fillna("None")

sns.factorplot("Fireplaces","SalePrice",data=raw_data,hue="FireplaceQu");
sns.factorplot()





11:14
import seaborn as sns

Number of Numerical features:  38
Number of Categorical features:  43