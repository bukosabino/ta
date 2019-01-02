import pandas as pd

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add adx indicator filling Nans values
df['ichi_a'] = ichimoku_a(df['High'], df['Low'], fillna=True)
df['ichi_b'] = ichimoku_b(df['High'], df['Low'], fillna=True)

import pdb; pdb.set_trace()

print(df.head(20))

print(df.tail(20))

import pdb; pdb.set_trace()
