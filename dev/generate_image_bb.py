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

plt.plot(df[40700:41000].Close)
plt.plot(df[40700:41000].volatility_bbh, label='High BB')
plt.plot(df[40700:41000].volatility_bbl, label='Low BB')
plt.plot(df[40700:41000].volatility_bbm, label='EMA BB')
plt.title('Bollinger Bands')
plt.legend()
plt.savefig("bb.png")
