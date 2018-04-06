import numpy as np
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add all ta features
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC")

# Fill nan values
df = df.fillna(method='backfill')

print(df.columns)
