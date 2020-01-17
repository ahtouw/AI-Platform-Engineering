# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:07:58 2020

@author: gmnya

This module is for Feature Engineering on the following 
features:
    BsmtUnfSF
    TotalBsmtSF
    1stFlrSF
    2ndFlrSF

This is done in the following function engineer(), which will
receive the original dataframe, and treat any missing data and 
outliers, and adjust for skewness to help make the data fit a 
normal distribution.

"""


import pandas as pd
import numpy as np

def basement_type(df_in:pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for:
        BsmtFinType1
        BsmtFinType2

    Parameters
    ----------
    df_in : pd.DataFrame
        This dataframe will only include ONLY the 
        above listed engineered features.

    Returns
    -------
    Complete dataframe with engineered features.

    """
    # Define Replacement Dictionaries
    BsmtLiving_Dict = {
        'GLQ':3,
        'ALQ':2,
        'BLQ':1,
        'Rec':np.NaN,
        'LwQ':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN
    }

    Rec_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'LwQ':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN,
        'Rec':1
    }
    
    LwQ_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'Rec':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN,
        'LwQ':1
    }

    Unf_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'Rec':np.NaN,
        'LwQ':np.NaN,
        'Na':np.NaN,
        'Unf':1
    }

    
    # Replace the incoming data based on the above dictionary. 
    # Leave NaN's - they're important
    BsmtFinT1Score = df_in.BsmtFinType1.replace(BsmtLiving_Dict)
    BsmtFinT2Score = df_in.BsmtFinType2.replace(BsmtLiving_Dict)

    BsmtRecT1Score = df_in.BsmtFinType1.replace(Rec_Dict)
    BsmtRecT2Score = df_in.BsmtFinType2.replace(Rec_Dict)

    BsmtLwQT1Score = df_in.BsmtFinType1.replace(LwQ_Dict)
    BsmtLwQT2Score = df_in.BsmtFinType2.replace(LwQ_Dict)

    BsmtUnfT1Score = df_in.BsmtFinType1.replace(Unf_Dict)
    BsmtUnfT2Score = df_in.BsmtFinType2.replace(Unf_Dict)

    # Now shuffle them together by replacing the NaN's in T1 with the values in T2.
    # We can also safely fill the remaining NaN's with zeroes, cast to integer, and
    # rename the series.
      
    BsmtFin = BsmtFinT1Score.fillna(BsmtFinT2Score).fillna(0).apply(int).rename('BsmtFin_Living')
    BsmtRec = BsmtRecT1Score.fillna(BsmtRecT2Score).fillna(0).apply(int).rename('BsmtFin_Rec')
    BsmtLwQ = BsmtLwQT1Score.fillna(BsmtLwQT2Score).fillna(0).apply(int).rename('BsmtFin_LwQ')
    BsmtUnf = BsmtUnfT1Score.fillna(BsmtUnfT2Score).fillna(0).apply(int).rename('BsmtFin_Unf')

    #Concatenate the resultant columns into a single dataFrame for return
    ret = pd.concat(
        [
            BsmtFin,
            BsmtRec,
            BsmtLwQ,
            BsmtUnf
        ], 
        axis=1
    
    )
    return ret


# this section is entirely for testing purposes. When 
# it is imported, it will not run.

def main():
    df_in=pd.read_csv('train.csv')
    print(basement_type(df_in))
if __name__ == '__main__':
    main()