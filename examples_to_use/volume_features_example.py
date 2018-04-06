"""
This is a example adding volume features. You can edit calling any function
in wrapper.py file:
https://github.com/bukosabino/ta/blob/master/ta/wrapper.py
"""

import numpy as np
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add all volume features
df = add_volume_ta(df, "High", "Low", "Close", "Volume_BTC")

# Clean nan values
df = df.fillna(method='backfill')

print(df.columns)
