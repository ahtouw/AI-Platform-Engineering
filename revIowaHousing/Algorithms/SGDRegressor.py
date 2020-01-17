# This algorithm was coded collaboratively between Team Lead Alex Buckalew and associates Hanan Kwok, William Ah Tou,
# Brandon Jackson, and Deward Seneh
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def sgd_model(df:pd.DataFrame, y:pd.Series) -> str:
    """SGD Regressor model that returns score."""
    # Checks if Foundation is still in the columns and if it is still set to string and drops it if it is
    if 'Foundation' in df.columns and df.Foundation.dtype == object:
        df.drop(axis=1, labels='Foundation', inplace=True)
    
    # Instantiates the MinMaxScaler and scales our dataframe before train/test split
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    # Train/Test data split
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

    # Create model with 10,000 iterations and warm_start set to true so it reuses previous iteration for initialization values
    model = SGDRegressor(max_iter=10000, warm_start=True)
    model.fit(x_train, y_train)
    return f'{model.score(x_test, y_test):.2f}'

if __name__ == '__main__':
    df = pd.read_csv('housemodel/fixed.csv')
    y = pd.read_csv('other/train.csv')['SalePrice']
    print(sgd_model(df, y))
