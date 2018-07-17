import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ta import *

# Load data
df = pd.read_csv('../data/datas.csv', sep=',')
df = utils.dropna(df)

# Add all ta features filling nans values
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC",
                            fillna=True)
# Generate macd image
plt.plot(df[40500:41000].trend_macd, label='MACD')
plt.plot(df[40500:41000].trend_macd_signal, label='MACD Signal')
plt.plot(df[40500:41000].trend_macd_diff, label='MACD Difference')
plt.title('MACD, MACD Signal and MACD Difference')
plt.legend()
plt.savefig("macd.png")
