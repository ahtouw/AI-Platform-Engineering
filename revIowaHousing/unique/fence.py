# Michael Grillo

import pandas as pd

def fence_uniq(df_in):
    df_out = pd.DataFrame()
    df_out['Fence_Wood'] = df_in.Fence.replace(["MnWw", "GdWo", "NA", "MnPrv", "GdPrv"], [1, 2, 0, 0, 0]).fillna(0)
    df_out['Fence_Private'] = df_in.Fence.replace(["MnWw", "GdWo", "NA", "MnPrv", "GdPrv"], [0, 0, 0, 1, 2]).fillna(0)
    return df_out
	

