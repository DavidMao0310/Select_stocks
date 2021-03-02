import pandas as pd
import numpy as np
from canslim import *

pd.set_option('display.max_columns', None)
# We need collect all stock data then make them into a single dataframe which is tracking by symbol
stocks = pd.read_csv('stocks_pred.csv')
stocks['Quarter end'] = pd.to_datetime(stocks['Quarter end'])


############################################################################
##CANSLIM---A
def Afilter(df, Earing_growth_per, Earing_growth_year_need, ROE_growth):
    '''

    :param df: dataframe contains 'Earnings' and 'ROE' columns
    :param Earing_growth_per: Annual earings growth should be at least $ percentage
    :param Earing_growth_year_need: Annual earings growth should be at least $ percentage over $ years
    :param ROE_growth: Annual Return on Equity should be more than $ percentage
    '''
    ac = AnnualChange(cols=['Earnings'], add_percent=True)
    df = ac.fit_transform(df)
    qc = QuarterlyChange(cols=['Earnings'])
    df = qc.fit_transform(df)
    si = StreakIncrease(cols=['Earnings_annualp'], x=Earing_growth_per)
    df = si.fit_transform(df)
    A_filter1 = (df['Earnings_annualp_streakgt_' + str(Earing_growth_per)] >= Earing_growth_year_need) & \
                (df['ROE'] > ROE_growth)

    return A_filter1


A_filter = Afilter(stocks, Earing_growth_per=0.25, Earing_growth_year_need=3, ROE_growth=0.17)

A_select_stocks = stocks[A_filter]


print(max(stocks['Quarter end']).year)
A_select_stocks_end = A_select_stocks[A_select_stocks['Year'] >= max(stocks['Quarter end']).year - 1]
A_select_stocks_end.reset_index(inplace=True)
print(A_select_stocks_end)
print(A_select_stocks_end['symbol'].drop_duplicates(keep=False).values)
