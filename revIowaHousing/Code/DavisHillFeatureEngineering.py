# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# initialize dataframe from data

def feature_extract(df_raw):
  # isolate target features and initialize new dataframe with target features

  lst = ['BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle']
  df = df_raw[lst]

  # convert categorical features to string

  df['OverallCond'] = df['OverallCond'].astype(str)
  df['OverallQual'] = df['OverallQual'].astype(str)

  # check for null values, there are none

  df.isnull().sum()

  # produce dummy variables from categorical variables

  df_dummies = pd.get_dummies(df,drop_first=True,prefix=lst)

  return df_dummies
