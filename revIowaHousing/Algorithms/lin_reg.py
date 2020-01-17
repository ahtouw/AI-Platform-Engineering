import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import linear_model

#Initial Implementation: Desmond
#Algorithm Analysis: John
#Refactoring/Final Implementation: Davis
#Data Analysis/Statistics: Corey, Ragy
#Algorithm Research: George A.

#Ragy, George A., Davis, Corey, John, Desmond

def lin_reg(df: pd.DataFrame, y: pd.DataFrame):
    '''Takes in a df and a target column. Splits for train/test data, runs a linear regression, and outputs metrics.'''
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, train_size = 0.8)
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    MAE = mean_absolute_error(y_test , predictions)
    r2 = r2_score(y_test, predictions)
    cdf = pd.DataFrame(lm.coef_, df.columns, columns = ['Coeff'])
    print('MAE: ', MAE, 'R2: ', r2, '\nCoefficients: ', cdf)
