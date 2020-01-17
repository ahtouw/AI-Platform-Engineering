#George Avitesyan

import pandas as pd

def conditions_2hot(df_in:pd.DataFrame) -> pd.DataFrame:
    '''
    
    return 2 hot encoder

    '''
    conditionlist = [
        'Artery',
        'Feedr',
        'Norm',
        'RRNn',
        'RRAn',
        'PosN',
        'PosA',
        'RRNe',
        'RRAe'
    ]
    df_out = pd.DataFrame()
    for x in conditionlist:
        colname = 'Cond_' + x
        df_out[colname] = ((df_in.Condition1 == x)|(df_in.Condition2 == x))
        df_out[colname] = df_out[colname].replace({False:0, True:1})
    return df_out.fillna(0)
