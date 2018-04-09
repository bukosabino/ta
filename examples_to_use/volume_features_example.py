"""This is a example adding volume features.
"""
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add all volume features filling nans values
df = add_volume_ta(df, "High", "Low", "Close", "Volume_BTC", fillna=True)

print(df.columns)
