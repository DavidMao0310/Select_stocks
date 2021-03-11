from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

def cal_ret(df, w):
    '''w:week5; month:20; semi-annual:120; annual:250
    '''
    df = df / df.shift(w) - 1
    return df.iloc[w:, :].fillna(0)



def get_RPS(ser):
    df = pd.DataFrame(ser.sort_values(ascending=False))
    df['n'] = range(1, len(df) + 1)
    df['rps'] = (1 - df['n'] / len(df)) * 100
    return df


def all_RPS(data):
    dates = data.index
    dates1 = [x for x in dates]
    st = dates[0]
    end = dates[-1]
    RPS = {}
    for i in range(len(data)):
        RPS[dates[i]] = pd.DataFrame(get_RPS(data.iloc[i]).values,
                                     columns=['Rate of Return', 'Rank', 'RPS'],
                                     index=get_RPS(data.iloc[i]).index)
    return RPS, dates1, st, end



def all_data(rps, ret):
    df = pd.DataFrame(np.NaN, columns=ret.columns, index=ret.index)
    for date in ret.index:
        d = rps[date]
        for c in d.index:
            df.loc[date, c] = d.loc[c, 'RPS']
    return df



def Lfilter(df):
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
    df.set_index('trade_date', inplace=True)
    df.index = (pd.to_datetime(df.index)).strftime('%Y-%m-%d')
    df.dropna(inplace=True, axis=1)

    window = 120
    ret_use = cal_ret(df, w=window)
    rps_use, rps_dates, rps_start, rps_end = all_RPS(ret_use)
    dates_list = [datetime.strftime(x, '%F') for x in pd.date_range(rps_start, rps_end, freq='m')]
    df_rps_rank = pd.DataFrame()

    for date in dates_list:
        while date not in rps_dates:
            dt = datetime.strptime(date, '%Y-%m-%d')
            date = dt - timedelta(days=1)
            date = date.strftime('%Y-%m-%d')
        df_rps_rank[date] = rps_use[date].index[:50]
    L_select_stocks = df_rps_rank.iloc[:, -1].values.tolist()
    # L select can get the strength candidates
    return L_select_stocks

