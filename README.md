# Technical Analysis Library in Python

You can use this library to add features to your trade dataset.

## Volume

* Accumulation/Distribution Index (ADI)
* On-balance volume (OBV)
* On-balance volume mean (OBV mean)
* Chaikin Money Flow (CMF)
* Force Index (FI)
* Ease of movement (EoM, EMV)
* Volume-price trend (VPT)

## Volatility

* Average True Range (ATR)
* Bollinger Bands (BB)
* Keltner channel (KC)
* Donchian channel (DC)


# Installation

> pip install -r requirements.txt


# Example add one feature

```python
import pandas as pd
from volume import *

# load datas
df = pd.read_csv('input/data.csv', sep=',')

# clean nan values
df = utils.dropna(df)

# add ta feature
df['cmf'] = chaikin_money_flow(df.High, df.Low, df.Close, df.Volume_BTC)

# fill nan values
df['cmf'] = df['cmf'].fillna(method='backfill')
```


# Example add some features

```python
import pandas as pd
from volume import *

# load datas
df = pd.read_csv('input/data.csv', sep=',')

# clean nan values
df = utils.dropna(df)

# add ta features
df = add_volume_ta(df, "High", "Low", "Close", "Volume_BTC")

# fill nan values
df = df.fillna(method='backfill')
```


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# Credits:

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

We are glad to receive any contribution, idea or feedback.