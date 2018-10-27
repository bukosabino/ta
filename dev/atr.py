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
df['adx'] = atr(df['High'], df['Low'], df['Close'], n=14, fillna=True)

import pdb; pdb.set_trace()

print(df['adx'])

print(df.columns)
