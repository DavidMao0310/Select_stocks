from canslim import *
import pandas as pd

pd.set_option('display.max_columns', None)
# We need collect all stock data then make them into a single dataframe which is tracking by symbol
stocks = pd.read_csv('stocks_pred.csv')
stocks['Quarter end'] = pd.to_datetime(stocks['Quarter end'])

def Cfilter(df, EPSgrowth_prequa, EPSgrowth_preyear_samequa, EPS_increasly_acc):
    '''

    :param df: dataframe contains 'EPS basic' columns
    :param EPSgrowth_prequa: EPS growth compared to previous quarters
    :param EPSgrowth_preyear_samequa: EPS growth compared to same quarters previous year
    :param EPS_increasly_acc: EPS increasly accelerating for at least quarters
    :return: CANSLIM_C-filter
    '''
    ac = AnnualChange(cols=['EPS basic'])
    df = ac.fit_transform(df)
    qc = QuarterlyChange(cols=['EPS basic'])
    df = qc.fit_transform(df)
    ag = AccelIncrease(cols=['EPS basic_qperc'])
    df = ag.fit_transform(df)
    # EPS basic: EPS value
    # EPS basic_qchange: Quartly change in EPS
    # EPS basic_annual: Annually change in EPS
    # EPS basic_qperc: EPS compared to previous quarters
    EPS_filter = (df['EPS basic_qperc'] >= EPSgrowth_prequa) & (df['EPS basic_annualp'] >= EPSgrowth_preyear_samequa) & \
                 (df['EPS basic_qperc_accstreak'] >= EPS_increasly_acc)
    return EPS_filter

C_filter = Cfilter(stocks, 0.2, 0.2, 2)

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
A_filter = Afilter(stocks, Earing_growth_per=0.2, Earing_growth_year_need=2, ROE_growth=0.17)

AC_filter = C_filter & A_filter
AC_select_stocks = stocks[AC_filter]


AC_select_stocks_end = AC_select_stocks[AC_select_stocks['Year'] >= max(stocks['Quarter end']).year - 1]
AC_select_stocks_end.reset_index(inplace=True)
print(AC_select_stocks_end)
print(AC_select_stocks_end['symbol'].drop_duplicates(keep=False).values)
