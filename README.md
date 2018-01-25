# Technical Analysis Library in Python

You can use this library to add features to your trade dataset.

## Volume

* Accumulation/Distribution Index (ADI)
* On-balance volume (OBV)
* On-balance volume mean (OBV)
* Chaikin Money Flow (CMF)
* Force Index (FI)
* Ease of movement (EoM, EMV)
* Volume-price trend (VPT)

# Installation

> pip install -r requirements.txt

# Use

```python
import pandas as pd
from volume import *
df = pd.read_csv('input/data.csv', sep=',')
df['cmf'] = chaikin_money_flow(df.High, df.Low, df.Close, df.Volume_BTC)
```


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# Credits:

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

We are glad receving any contribution, idea or feedback.
