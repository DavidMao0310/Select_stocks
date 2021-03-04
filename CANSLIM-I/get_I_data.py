import pandas as pd

pd.set_option('display.max_columns', None)
from finvizfinance.screener.ownership import Ownership

foverview = Ownership()
filters_dict = {'Exchange': 'NASDAQ'}
foverview.set_filter(filters_dict=filters_dict)
df = foverview.ScreenerView()
print(df)
# df.to_csv('ownership(NASDAQ).csv',index=False)
