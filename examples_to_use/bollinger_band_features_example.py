"""This is a example adding bollinger band features.
"""

import numpy as np
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add bollinger band high indicator
df['bb_high_indicator'] = bollinger_hband_indicator(df["Close"], n=20, ndev=2)

# Add bollinger band low indicator
df['bb_low_indicator'] = bollinger_lband_indicator(df["Close"], n=20, ndev=2)

# Clean nan values
df = df.fillna(method='backfill')

print(df.columns)
