# Technical Analysis Library in Python

You can use this library to add features to your financial time series dataset.

### Volume

* Accumulation/Distribution Index (ADI)
* On-Balance Volume (OBV)
* On-Balance Volume mean (OBV mean)
* Chaikin Money Flow (CMF)
* Force Index (FI)
* Ease of Movement (EoM, EMV)
* Volume-price Trend (VPT)

### Volatility

* Average True Range (ATR)
* Bollinger Bands (BB)
* Keltner Channel (KC)
* Donchian Channel (DC)

### Trend

* Moving Average Convergence Divergence (MACD)
* Average Directional Movement Index (ADX)
* Vortex Indicator (VI)
* Trix (TRIX)
* Mass Index (MI)
* Commodity Channel Index (CCI)
* Detrended Price Oscillator (DPO)
* KST Oscillator (KST)
* Ichimoku Kinkō Hyō (Ichimoku)

### Momentum

* Money Flow Index (MFI)
* Relative Strength Index (RSI)

### Others

* Daily Return (DR)
* Cumulative Return (CR)


# How to use


```sh
> virtualenv -p python3 virtualenvironment
> source virtualenvironment/bin/activate
> pip3 install ta
```

You can get code examples in "examples_to_use" folder.

If you don't know any feature you can view the notebook: "examples/visualize_features.ipynb".

Note: To execute the notebook you will need install 'matplotlib' and 'jupyter lab'. So:

```sh
> pip3 install matplotlib==2.1.2
> pip3 install jupyterlab==0.31.12
```


### Example adding all features

```python
import pandas as pd
from ta import *

# load datas
df = pd.read_csv('your-file.csv', sep=',')

# clean nan values
df = utils.dropna(df)

# add ta features
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC")

# fill nan values
df = df.fillna(method='backfill')
```


### Example adding individual features

```python
import pandas as pd
from ta import *

# load datas
df = pd.read_csv('your-file.csv', sep=',')

# clean nan values
df = utils.dropna(df)

# Add bollinger band high indicator
df['bb_high_indicator'] = bollinger_hband_indicator(df["Close"], n=20, ndev=2)

# Add bollinger band low indicator
df['bb_low_indicator'] = bollinger_lband_indicator(df["Close"], n=20, ndev=2)

# fill nan values
df['cmf'] = df['cmf'].fillna(method='backfill')
```


# Deploy to developers

```sh
> pip3 install -r requirements.txt
```


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# TODO:

* extend documentation
* add ta features
* fillna by function (boolean parameter)


# Credits:

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

We are glad to receive any contribution, idea or feedback.
