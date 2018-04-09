"""This is a example adding all technical analysis features implemented in
this library.
"""
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add all ta features filling nans values
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC",
                                fillna=True)

print(df.columns)
print(len(df.columns))
