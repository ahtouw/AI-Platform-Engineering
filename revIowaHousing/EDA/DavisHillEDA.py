#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df_raw = pd.read_csv("c:/Users/Alfred/house_prices/train.csv")
lst = ['BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle','SalePrice']
df = df_raw[lst]
df.head()
sns.violinplot(x="HouseStyle",y="SalePrice",data=df)
sns.swarmplot(x='BldgType',y='SalePrice',data=df,size=3)
sns.swarmplot(x='HouseStyle',y='SalePrice',data=df,size=3)
sns.stripplot(x="HouseStyle",y="SalePrice",data=df)
sns.boxplot(x='HouseStyle',y='SalePrice',data=df)
sns.violinplot(x="OverallCond",y="SalePrice",data=df)
sns.swarmplot(x='OverallCond',y='SalePrice',data=df)
sns.stripplot(x="OverallCond",y="SalePrice",data=df)
sns.boxplot(x='OverallCond',y='SalePrice',data=df)
sns.violinplot(x="OverallQual",y="SalePrice",data=df)
sns.swarmplot(x='OverallQual',y='SalePrice',data=df)
sns.stripplot(x="OverallQual",y="SalePrice",data=df)
sns.boxplot(x='OverallQual',y='SalePrice',data=df)
sns.scatterplot(x="SalePrice",y="OverallQual",data=df)
sns.boxplot(x="OverallQual",y="OverallCond",data=df)
corr_mat = df.corr()
sns.violinplot(x="OverallQual", y="SalePrice", data=df,palette='rainbow')
sns.swarmplot(x="OverallQual", y="SalePrice", data=df,color='black',size=5)
sns.violinplot(x="OverallCond", y="SalePrice", data=df,palette='rainbow')
sns.swarmplot(x="OverallCond", y="SalePrice", data=df,color='blue',size=7)

sns.countplot(x='OverallCond',data=df)

sns.countplot(x='OverallQual',data=df)

sns.countplot(x='BldgType',data=df)

sns.countplot(x='HouseStyle',data=df)


sns.countplot(x='RoofStyle',data=df)


