import pandas as pd
import numpy as np
from canslim import *

pd.set_option('display.max_columns', None)
# We need collect all stock data then make them into a single dataframe which is tracking by symbol
stocks = pd.read_csv('stocks_pred.csv')
stocks['Quarter end'] = pd.to_datetime(stocks['Quarter end'])


############################################################################
##CANSLIM---C
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


C_filter = Cfilter(stocks, 0.2, 0.25, 3)

C_select_stocks = stocks[C_filter]
print(max(stocks['Quarter end']).year)
C_select_stocks_end = C_select_stocks[C_select_stocks['Year'] >= max(stocks['Quarter end']).year - 1]
C_select_stocks_end.reset_index(inplace=True)
print(C_select_stocks_end)
print(C_select_stocks_end['symbol'].drop_duplicates(keep=False).values)
