import pandas as pd

pd.set_option('display.max_columns', None)

stocks = pd.read_csv('ownership(NASDAQ).csv')
stocks = stocks[['Ticker', 'Inst Own']]
stocks.dropna(inplace=True)
I_filter = stocks['Inst Own'] > 0.8
I_select_stocks = stocks[I_filter]['Ticker'].values
print(I_select_stocks)
