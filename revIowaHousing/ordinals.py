### THIS IS WHERE WE CLEAN UP ORDINAL AND UNIQUE FEATURES
import pandas as pd
from ordinalRepl import ordinalRepl

def ordinals(df) -> pd.DataFrame:

    ### Dictionaries and fill values
    ## Put your feature’s information here in the below-outlined format.
    ## Leave a space after each feature.

    ## BsmtExposure - Kyle Cloud
    BsmtExposure_dict = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No':1, 'NA':0}
    BsmtExposure_fillval = 0

    ## Functional - John Winegardner
    Functionaldict = {
        "Typ": 0,
        "Min1": 1,
        "Min2": 2,
        "Mod": 3,
        "Maj1": 4,
        "Maj2": 5,
        "Sev": 6,
        "Svg": 7
    }
    Functional_fill = 0

    # FireplaceQu - John Winegardner
    FireplaceQudict = {
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5,
        "NA": 0
    }
    FireplaceQu_fill = 0

    # GarageFinish - John Winegardner
    GarageFinishdict = {
        "Unf": 1,
        "RFn": 2,
        "Fin": 3
    }
    GarageFinish_fill = 0
    
    # LandSlope - Brandon Jackson
    landSlope_dict = {'Sev':3, 'Mod':2, 'Gtl':1}
    landSlope_fill = 1

    # The Rest - Deward Seneh
    generic_dict = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}
    generic_list = ['BsmtCond','BsmtQual','GarageQual','GarageCond','PoolQC','ExterQual','KitchenQual','HeatingQC','ExterCond']
    generic_filler = 0

    ### Standard Ordinal Replacement Calls
    df_out = pd.concat(
        [
            ordinalRepl(df.BsmtExposure, BsmtExposure_dict, BsmtExposure_fillval),
            ordinalRepl(df.Functional, Functionaldict, Functional_fill),
            ordinalRepl(df.FireplaceQu, FireplaceQudict, FireplaceQu_fill),
            ordinalRepl(df.GarageFinish, GarageFinishdict, GarageFinish_fill),
            ordinalRepl(df.LandSlope, landSlope_dict, landSlope_fill)
            #insert your function call above this comment
        ],
        axis = 1  
    )
    df.drop(['BsmtExposure', 'Functional', 'FireplaceQu', 'GarageFinish', 'LandSlope'], axis=1, inplace=True)

    # Concatenating the generics - Stephen Wight
    df_generic = pd.concat([ordinalRepl(df[x], generic_dict, generic_filler) for x in generic_list], axis = 1)
    df_out = pd.concat([df_out, df_generic], axis = 1)
    df.drop(generic_list, axis=1, inplace=True)

    df = pd.concat([df, df_out], axis=1)

    df_obj = df.loc[:,df.dtypes == 'object']
    df_num = df.loc[:,~(df.dtypes == 'object')]
    
    return df_num, df_obj
	

if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	# print(ordinals(df).head())


