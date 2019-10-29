"""This is a example adding all technical analysis features implemented in
this library.
"""
import pandas as pd

import sys
sys.path.append("..") # Adds higher directory to python modules path.
import ta

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = ta.utils.dropna(df)

print(df.columns)

# Add all ta features filling nans values
df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC",
                                fillna=True)

# Add all ta features not filling nans values
df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC",
                                fillna=False)

print(df.columns)
print(len(df.columns))
