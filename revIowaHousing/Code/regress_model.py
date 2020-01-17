#Additional assistance for this file came from the OLS group: Corey, George A., John, Ragy, and Davis

'''Each of the model functions drops the SalePrice column so do not run them sequentially. '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import linear_model

def split(df):
    '''df is the x. y is the target. Returns a tuple, so slice by index (split(df)[0] returns x_train.)'''
    my_column = 'SalePrice' #input("What would you like to predict? ")
    y = df[my_column]
    df.drop([my_column], axis = 1, inplace = True)
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, train_size = 0.8)
    return (x_train, x_test, y_train, y_test)

def lin_reg(df:pd.DataFrame):
    x_train, x_test, y_train, y_test = split(df)
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    MAE = mean_absolute_error(y_test , predictions)
    r2 = r2_score(y_test, predictions)
    cdf = pd.DataFrame(lm.coef_, df.columns, columns = ['Coeff'])
    return (MAE, r2, cdf)


def lasso(df):
    x_train, x_test, y_train, y_test = split(df)
    lm = linear_model.Lasso(fit_intercept = True, normalize = True, max_iter=10000)
    lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    MAE = mean_absolute_error(y_test , predictions)
    r2 = r2_score(y_test, predictions)
    return (MAE, r2)
