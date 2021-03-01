import pandas as pd
import numpy as np
from canslim import *
pd.set_option('display.max_columns',None)
#We need collect all stock data then make them into a single dataframe which is tracking by symbol
stocks = pd.read_csv('stocks_test.csv')
stocks = stocks.iloc[:,1:]
stocks = stocks.replace('None',np.NAN)
#print(stocks.info())

def C_Preprocessing(df):
    # Convert every column other than symbol or Quarter end to float.
    df[df.columns.difference(['symbol', 'Quarter end'])] = df[
        df.columns.difference(['symbol', 'Quarter end'])].astype(float)
    # Convert Quarter end to datetime
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    # Generate a Year and Month column from Quarter end
    df['Year'] = df['Quarter end'].dt.year
    df['Month'] = df['Quarter end'].dt.month
    # save original columns in case needed later
    og_columns = df.select_dtypes(include=['float']).columns
    round(df.isnull().sum() / len(df) * 100, 2)

    # Generating Price Target columns. Both total change(Price target) and percentage change(Pricep target)
    pt = PriceTarget()
    df = pt.fit_transform(df)

    # Fill the large outliers with the NaN values
    df.loc[df[df['Pricep target'] > 2].index - 1, 'Price'] = np.nan
    df.loc[list(df[df['Pricep target'] > 2].index - 1) + \
               list(df[df['Pricep target'] > 2].index) + \
               list(df[df['Pricep target'] > 2].index + 1)] \
        .sort_values(['symbol', 'Year', 'Month']) \
        ['Price'] = df.loc[list(df[df['Pricep target'] > 2].index - 1) + \
                               list(df[df['Pricep target'] > 2].index) + \
                               list(df[df['Pricep target'] > 2].index + 1)] \
        .sort_values(['symbol', 'Year', 'Month']) \
        ['Price'].fillna(method='ffill')

    # Regenerate our target columns after cleaning price data
    pt = PriceTarget()
    df = pt.fit_transform(df)
    return df

stocks=C_Preprocessing(stocks)
print(stocks)


############################################################################
stocks.to_csv('stocks_pred.csv',index=False)