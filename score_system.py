import pandas as pd

pd.set_option('display.max_columns', None)
import tushare as ts
from datetime import datetime

token = 'Type Your Token'
ts.set_token(token)
pro = ts.pro_api(token)

stocklist = pro.stock_basic(exchange='', list_status='L',
                            fields='ts_code,symbol,enname,area,industry,list_date')
stocklist = stocklist[stocklist['list_date'].apply(int).values < 20210101][:1]
code_name = dict(zip(stocklist.enname.values, stocklist.ts_code.values))


def get_indicators(code):
    t0 = datetime.now()
    t1 = datetime(t0.year - 4, t0.month, t0.day)
    end = t0.strftime('%Y%m%d')
    start = t1.strftime('%Y%m%d')
    fields = 'ann_date,end_date,tr_yoy,op_yoy,\
         grossprofit_margin,expense_of_sales,inv_turn,eps,\
         ocfps,roe_yearly,roa2_yearly,netprofit_yoy'
    fina = (pro.fina_indicator(ts_code=code, start_date=start, end_date=end, fields=fields)
            .drop_duplicates(subset=['ann_date', 'end_date'], keep='first'))
    fina.set_index('end_date', inplace=True)
    fina = fina.sort_index()
    pbe = pro.daily_basic(ts_code=code, fields='trade_date,pe_ttm,pb')
    pbe.set_index('trade_date', inplace=True)
    pbe = pbe.sort_index()
    df = pd.merge(fina, pbe, left_index=True, right_index=True, how='left')
    df['pb'].fillna(method='ffill', inplace=True)
    return df


data_dict = {}
for name, code in code_name.items():
    try:
        data_dict[code] = get_indicators(code)
    except:
        pass


# print(data_dict)

# All score is scaled to 0 to 10
# There is a missing value or negative value (P/E ratio) in the case of the score is directly 0
def cal_tryoy(y):
    '''y is growth rate of revenue'''
    try:
        return 5 + min(round(y - 10), 5) if y >= 10 else 5 + max(round(y - 10), -5)
    except:
        return 0


def cal_opyoy(y):
    '''y is growth rate of operating profit'''
    try:
        return 5 + min(round((y - 20) / 2), 5) if y >= 20 else 5 + max(round((y - 20) / 2), -5)
    except:
        return 0


# Similar to CANSLIM Method
def cal_gpm(y):
    '''y is the most recent quarter gross margin - the average gross margin of the previous three quarters'''
    try:
        return 5 + min(round(y) / 0.5, 5) if y > 0 else max(round(y) / 0.5, -5) + 5
    except:
        return 0


def cal_exp(y):
    '''y is the most recent quarter expense ratio - the average period expense ratio of the previous three quarters'''
    try:
        return 5 + min(round(y) / 0.5, 5) if y > 0 else max(round(y) / 0.5, -5) + 5
    except:
        return 0


def cal_inv(y):
    '''y is (Most Recent Quarter Inventory Turnover - Average Inventory Turnover of Previous Three Quarters)/(Average Inventory Turnover of Previous Three Quarters) *100'''
    try:
        return 5 + min(round(y / 2), 5) if y > 0 else max(round(y / 2), -5) + 5
    except:
        return 0


# Operating cash flow per share score
def cal_ocfp(y):
    '''y is (sum of operating cash flow per share for the latest three quarters - sum of earnings per share for the latest three quarters)/(sum of earnings per share for the latest three quarters) *100'''
    try:
        return 5 + min(round(y / 4), 5) if y > 0 else max(round(y / 4), -5) + 5
    except:
        return 0


# ROE score
# Come from CANSLIM
def cal_roe(y):
    '''y is the annualized return on equity'''
    try:
        return 5 + min(round(y - 15), 5) if y >= 15 else 5 + max(round(y - 15), -5)
    except:
        return 0


# ROA score
def cal_roa(y):
    '''y is the most recent quarterly annualized rate of return on total assets'''
    try:
        return min(round((y - 5) / 0.5), 10) if y >= 5 else max(round(y - 5), 0)
    except:
        return 0


# P/B ratio score
def cal_pb(y):
    '''y is P/B ratio'''
    try:
        return 5 - max(round((y - 3) / 0.4), -5) if y <= 3 else 5 - min(round((y - 3) / 0.4), 5)
    except:
        return 0


# the moving P/E ratio relative to earnings growth(PEG ratio)
def cal_pe(y):
    '''y is PEG ratio'''
    try:
        return 5 - max(round((y - 1) / 0.1), -5) if y <= 1 else 5 - min(round((y - 1) / 0.1), 5)
    except:
        return 0


def indicator_score(df):
    data = df
    '''(1)Growth rate of revenue score'''
    data['SCORE growth rate of revenue'] = data['tr_yoy'].apply(cal_tryoy)
    '''(2)Operating profit growth rate score'''
    data['SCORE operating profit growth rate score'] = data['op_yoy'].apply(cal_opyoy)
    '''(3)Gross profit score'''
    # most recent - previous 3 quarters
    data['gpm'] = data['grossprofit_margin'] - data['grossprofit_margin'].rolling(3).mean()
    data['SCORE Gross profit'] = data['gpm'].apply(cal_gpm)
    '''(4)Period expense ratio score'''
    # most recent - previous 3 quarters
    data['exp'] = data['expense_of_sales'] - data['expense_of_sales'].rolling(3).mean()
    data['SCORE period expense'] = data['exp'].apply(cal_exp)
    '''(5)Turnover score'''
    # Find y
    data['inv'] = (data['inv_turn'] - data['inv_turn'].rolling(3).mean()) * 100 / data['inv_turn'].rolling(3).mean()
    data['SCORE Turnover rate'] = data['inv'].apply(cal_inv)
    '''(6)Operating cash flow per share score'''
    # Find y
    data['ocf'] = (data['ocfps'].rolling(3).sum() - data['eps'].rolling(3).sum()) * 100 / data['eps'].rolling(3).sum()
    data['SCORE Cash per share'] = data['ocf'].apply(cal_ocfp)
    '''(7)ROE score'''
    data['SCORE ROE'] = data['roe_yearly'].apply(cal_roe)
    '''(8)ROA score'''
    data['SCORE ROA'] = data['roa2_yearly'].apply(cal_roa)
    '''(9)P/B ratio score'''
    data['SCORE P/B ratio'] = data['pb'].apply(cal_pb)
    '''(10)PEG score'''
    # get PEG
    data['peg'] = data['pe_ttm'] / data['netprofit_yoy'].rolling(3).mean()
    data['SCORE PEG'] = data['peg'].apply(cal_pe)
    data['TOTAL'] = data[
        ['SCORE growth rate of revenue', 'SCORE operating profit growth rate score', 'SCORE Gross profit',
         'SCORE period expense',
         'SCORE Turnover rate', 'SCORE Cash per share', 'SCORE ROE',
         'SCORE ROA', 'SCORE P/B ratio', 'SCORE PEG']].sum(axis=1)
    return data[['SCORE growth rate of revenue', 'SCORE operating profit growth rate score', 'SCORE Gross profit',
                 'SCORE period expense',
                 'SCORE Turnover rate', 'SCORE Cash per share', 'SCORE ROE',
                 'SCORE ROA', 'SCORE P/B ratio', 'SCORE PEG', 'TOTAL']]


score_dict = {}
for i in data_dict.keys():
    score_dataframe = indicator_score(data_dict.get(i))
    score_dict[str(i)] = score_dataframe

print(score_dict)
