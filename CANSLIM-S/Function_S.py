import pandas as pd
from datetime import timedelta, datetime
import numpy as np

stocks = pd.read_csv('(vol.szsh)daily_data.csv')


def Sfilter(df):
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
    df_monthly = df.resample('M', on='trade_date').mean()
    df_monthly_pct1 = df_monthly.pct_change(1)
    # df_monthly_pct2=df_monthly.pct_change(2)
    df_monthly_incr = np.sign(df_monthly.diff(1)).rolling(window=3).sum()
    incr_filter = df_monthly_incr.iloc[-1] == 3

    df.set_index('trade_date', inplace=True)
    df.index = (pd.to_datetime(df.index)).strftime('%Y-%m-%d')
    end_date = df.index.values[-1]
    dt = datetime.strptime(end_date, '%Y-%m-%d')
    end_date = dt - timedelta(weeks=12)
    end_date = end_date.strftime('%Y-%m-%d')
    df = df[end_date:]
    df.dropna(inplace=True, axis=1)

    rolling_filter = df.iloc[-1] > df.mean() * 1.4
    month_filter1 = df_monthly_pct1.iloc[-1] > 0.4
    # month_filter2 = df_monthly_pct2.iloc[-1]>0.2

    return month_filter1 & rolling_filter & incr_filter


# monthly increase at least 40% compared to the previous month
# daily(target) increase at least 40% compared to the average of last 12 weeks
# There should be a continuously increase for 3 months

S_filter = Sfilter(stocks)
S_select_stocks = stocks.iloc[-1][S_filter].index.tolist()

print(S_select_stocks)
