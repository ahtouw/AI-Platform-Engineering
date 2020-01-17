# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:04:37 2019

@author: hanan

Make a Python function that takes in the whole Dataframe,
processes it, and returns a Dataframe with your features in it.
Avoid deleting rows from the dataset.
"""

import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# Key:
# SalePrice -> price of the house
# GarageCars -> size of the garage in terms of car capacity
# GarageArea -> size of the garage in square feet
# OpenPorchSF -> size of open porch area in square feet
# WoodDeckSF -> size of wood deck area in square feet

# Notes: Comment/uncomment print & plot lines of code as necessary to 
# view the data & results as it gets modified/cleaned

# Another note: Normalize & handle skewness in numerical variables and 
# do one-hotting in categorical variables

# Load the dataset
HPdf = pd.read_csv('C:/Users/hanan/Desktop/BatchProject/Dataset/train.csv')

# Perform univariate analysis on GarageCars, GarageArea, OpenPorchSF, WoodDeckSF
# Perform bivariate analysis between the 4 chosen features & the target feature SalePrice

# Set target feature: SalePrice
target = HPdf['SalePrice']

# Selecting features: GarageCars, GarageArea, WoodDeckSF, OpenPorchSF
assigned = HPdf[['SalePrice', 'GarageArea', 'GarageCars', 'OpenPorchSF', 'WoodDeckSF']]

# Data Exploration
#print("Info about the full dataset:\n")
#print(HPdf.info())
#print("\nShape of the full dataset:\n")
#print(HPdf.shape)
#print("\nInfo about the selected columns:\n")
#print(assigned.info())
#print("\nShape of the selected columns:\n")
#print(assigned.shape)

# Handling missing values in the data
for col in assigned.columns.values:
    percentage_total = assigned[col].sum() * 0.3 # set a threshold for the acceptable percentage of missing values in a column (usually 25 or 30 percent)
    missval = assigned[col].isnull().sum()
    if missval > percentage_total:
        assigned = assigned.drop(col, axis = 1)
    else:
        assigned = assigned.fillna(assigned[col].mean())   

# Handling outliers in the data -> I filtered for each feature sequentially.
# Note: I visualized the 5 chosen features to look for outliers
# before cleaning the data

# Filter outliers based on SalePrice 
#spfiltered = assigned[assigned['SalePrice'] <= 700000]
#print(spfiltered.shape)

# Filter outliers based on GarageArea
#gafiltered = spfiltered[spfiltered['GarageArea'] <= 1300]
#print(gafiltered.shape)

# Filter outliers based on Garage Cars
#gcfiltered = gafiltered[gafiltered['GarageCars'] < 4]
#print(gcfiltered.shape)

# Filter outliers based on OpenPorchSF
#opfiltered = gcfiltered[gcfiltered['OpenPorchSF'] <= 450]
#print(opfiltered.shape)

# Filter outliers based on WoodDeckSF
#wdfiltered = opfiltered[opfiltered['WoodDeckSF'] < 800]
#print(wdfiltered.shape)

import numpy as np
from scipy.stats import skew
cols = ['GarageArea', 'GarageCars', 'OpenPorchSF', 'WoodDeckSF']
myDF = assigned[cols]

#for col in cols:
#    mean = myDF[col].mean()
#    maxval = 3 * myDF[col].std() + mean
#    minval = -3 * myDF[col].std() + mean
#    myDF[col].clip(minval, maxval)    

for ga in myDF['GarageArea']: # Garage area in square feet
    if (ga >= 1300):
        myDF.replace(to_replace = ga, value = 1300)
        
for gc in myDF['GarageCars']: # Garage car capacity
    if (gc >= 4):
        myDF['GarageCars'].replace(to_replace = gc, value = 3)
            
for op in myDF['OpenPorchSF']: # Open porch area in square feet
    if (op > 450):
        myDF['OpenPorchSF'].replace(to_replace = op, value = 440)
            
for wd in myDF['WoodDeckSF']: # Wood deck area in square feet
    if (wd >= 800):
        myDF['WoodDeckSF'].replace(to_replace = wd, value = 790)

feat_skew = myDF.apply(skew)
feat_skew = feat_skew[feat_skew > 0.75]
loggedDF = np.log(myDF[feat_skew.index] + 1)
myDF = myDF.drop(feat_skew.index, axis = 1)
myDF = pd.concat([myDF, loggedDF], axis = 1)
        
print("GarageArea skew: %s\n" % myDF['GarageArea'].skew())
print("GarageCars skew: %s\n" % myDF['GarageCars'].skew())
print("OpenPorchSF skew: %s\n" % myDF['OpenPorchSF'].skew())
print("WoodDeckSF skew: %s\n" % myDF['WoodDeckSF'].skew())
        
#sns.distplot(myDF['GarageArea'], hist = True)
#sns.distplot(myDF['GarageCars'], hist = True)
#sns.distplot(myDF['OpenPorchSF'], hist = True)
#sns.distplot(myDF['WoodDeckSF'], hist = True)

# Assigning each column to its own variable
#sp = wdfiltered['SalePrice']
#ga = wdfiltered['GarageArea']
#gc = wdfiltered['GarageCars']
#op = wdfiltered['OpenPorchSF']
#wd = wdfiltered['WoodDeckSF']
#print("GarageArea skewness and kurtosis: %s %s" % (ga.skew(), ga.kurt()))
#print("GarageCars skewness and kurtosis: %s %s" % (gc.skew(), gc.kurt()))
#print("OpenPorchSF skewness and kurtosis: %s %s" % (op.skew(), op.kurt()))
#print("WoodDeckSF skewness and kurtosis: %s %s" % (wd.skew(), wd.kurt()))

# Univariate Analysis of each feature

# SalePrice UA -> requires log transformation to fit into normal distribution because of left skew
#sns.distplot(sp, hist = True) # SalePrice Histogram before log transformation
#sp_log = np.log(sp) # apply log transformation to SalePrice
#sns.distplot(sp_log, hist = True) # SalePrice Histogram after log transformation

# GarageArea UA
#sns.distplot(ga, hist = True)

# GarageCars UA
#sns.distplot(gc, hist = True)

# OpenPorchSF UA
#sns.distplot(op, hist = True)

# WoodDeckSF UA
#sns.distplot(wd, hist = True)

# Bivariate Analysis between SalePrice and each feature
colors = 'black'

# Correlation matrix with SalePrice and the 4 features
#corr = wdfiltered.corr()
#sns.heatmap(corr, cmap = 'coolwarm', annot = True)

# Scatter plot with SalePrice & GarageArea
#plt.figure(figsize = (20, 20))
#plt.scatter(ga, sp, c = colors)
#plt.title("House Price vs. Garage Area")
#plt.xlabel("Garage Area (in sq. ft.)")
#plt.ylabel("House Price")
#plt.yticks(np.arange(0, 700000, 50000))
#plt.show()

# Scatter plot with SalePrice & GarageCars
#plt.figure(figsize = (20, 20))
#plt.scatter(gc, sp, s = 10, c = colors)
#plt.title("House Price vs. Garage Car Capacity")
#plt.xlabel("Garage Car Capacity")
#plt.ylabel("House Price")
#plt.xticks(np.arange(0, 5, 1))
#plt.yticks(np.arange(0, 700000, 50000))
#plt.show()

# Scatter plot with SalePrice & OpenPorchSF
#plt.figure(figsize = (20, 20))
#plt.scatter(op, sp, c = colors)
#plt.title("House Price vs. Open Porch Area")
#plt.xlabel("Open Porch Area (in sq. ft.)")
#plt.ylabel("House Price")
#plt.yticks(np.arange(0, 700000, 50000))
#plt.show()

# Scatter plot with SalePrice & WoodDeckSF
#plt.figure(figsize = (20, 20))
#plt.scatter(wd, sp, c = colors)
#plt.title("House Price vs. Wood Deck Area")
#plt.xlabel("Wood Deck Area (in sq. ft.)")
#plt.ylabel("House Price")
#plt.yticks(np.arange(0, 700000, 50000))
#plt.show()