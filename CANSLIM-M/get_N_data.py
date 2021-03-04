import pandas as pd
from finvizfinance.quote import finvizfinance

pd.set_option('display.max_columns', None)

stocks = pd.read_csv('ownership(NASDAQ).csv')
stocks = stocks[['Ticker', 'Inst Own']]
stocks.dropna(inplace=True)
I_filter = stocks['Inst Own'] > 0.9
I_select_stocks = stocks[I_filter]['Ticker'].values
print(len(I_select_stocks), I_select_stocks)
full_news_list = []

for i in I_select_stocks:
    pstock = finvizfinance(str(i))
    news_df = pstock.TickerNews()
    news_df['symbol'] = str(i)
    print(str(i))
    full_news_list.append(news_df)

full_news = pd.concat(full_news_list)
print(full_news)
# full_news.to_csv('news.csv', index=False)
