![CircleCI](https://img.shields.io/circleci/build/github/bukosabino/ta/master)
[![Documentation Status](https://readthedocs.org/projects/technical-analysis-library-in-python/badge/?version=latest)](https://technical-analysis-library-in-python.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/bukosabino/ta/badge.svg)](https://coveralls.io/github/bukosabino/ta)
![PyPI](https://img.shields.io/pypi/v/ta)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ta)
[![Donate PayPal](https://img.shields.io/badge/Donate%20%24-PayPal-brightgreen.svg)](https://www.paypal.me/guau/3)

# Technical Analysis Library in Python

It is a Technical Analysis library useful to do feature engineering from financial time series datasets (Open, Close, High, Low, Volume). It is built on Pandas and Numpy.

![Bollinger Bands graph example](doc/figure.png)

The library has implemented 34 indicators:

#### Volume

* Accumulation/Distribution Index (ADI)
* On-Balance Volume (OBV)
* Chaikin Money Flow (CMF)
* Force Index (FI)
* Ease of Movement (EoM, EMV)
* Volume-price Trend (VPT)
* Negative Volume Index (NVI)
* Volume Weighted Average Price (VWAP)

#### Volatility

* Average True Range (ATR)
* Bollinger Bands (BB)
* Keltner Channel (KC)
* Donchian Channel (DC)

#### Trend

* Moving Average Convergence Divergence (MACD)
* Average Directional Movement Index (ADX)
* Vortex Indicator (VI)
* Trix (TRIX)
* Mass Index (MI)
* Commodity Channel Index (CCI)
* Detrended Price Oscillator (DPO)
* KST Oscillator (KST)
* Ichimoku Kinkō Hyō (Ichimoku)
* Parabolic Stop And Reverse (Parabolic SAR)

#### Momentum

* Money Flow Index (MFI)
* Relative Strength Index (RSI)
* True strength index (TSI)
* Ultimate Oscillator (UO)
* Stochastic Oscillator (SR)
* Williams %R (WR)
* Awesome Oscillator (AO)
* Kaufman's Adaptive Moving Average (KAMA)
* Rate of Change (ROC)

#### Others

* Daily Return (DR)
* Daily Log Return (DLR)
* Cumulative Return (CR)


# Documentation

https://technical-analysis-library-in-python.readthedocs.io/en/latest/


# Motivation to use

* [English](https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543)
* [Spanish](https://medium.com/datos-y-ciencia/biblioteca-de-an%C3%A1lisis-t%C3%A9cnico-sobre-series-temporales-financieras-para-machine-learning-con-cb28f9427d0)


# How to use (Python 3)

```sh
$ pip install --upgrade ta
```

To use this library you should have a financial time series dataset including `Timestamp`, `Open`, `High`, `Low`, `Close` and `Volume` columns.

You should clean or fill NaN values in your dataset before add technical analysis features.

You can get code examples in [examples_to_use](https://github.com/bukosabino/ta/tree/master/examples_to_use) folder.

You can visualize the features in [this notebook](https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb).


#### Example adding all features

```python
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna


# Load datas
df = pd.read_csv('ta/tests/data/datas.csv', sep=',')

# Clean NaN values
df = dropna(df)

# Add all ta features
df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC")
```


#### Example adding particular feature

```python
import pandas as pd
from ta.utils import dropna
from ta.volatility import BollingerBands


# Load datas
df = pd.read_csv('ta/tests/data/datas.csv', sep=',')

# Clean NaN values
df = dropna(df)

# Initialize Bollinger Bands Indicator
indicator_bb = BollingerBands(close=df["Close"], n=20, ndev=2)

# Add Bollinger Bands features
df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()

# Add Bollinger Band high indicator
df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

# Add Width Size Bollinger Bands
df['bb_bbw'] = indicator_bb.bollinger_wband()

# Add Percentage Bollinger Bands
df['bb_bbp'] = indicator_bb.bollinger_pband()
```


# Deploy and develop (for developers)

```sh
$ git clone https://github.com/bukosabino/ta.git
$ cd ta
$ pip install -r play-requirements.txt
$ make test
```


# Sponsor

![Logo OpenSistemas](doc/logo_neuroons_byOS_blue.png)

Thank you to [OpenSistemas](https://opensistemas.com)! It is because of your contribution that I am able to continue the development of this open source library.


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://pandas.pydata.org
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# In Progress:

* Automated tests for all the indicators.


# TODO:

* Use [NumExpr](https://github.com/pydata/numexpr) to speed up the NumPy/Pandas operations? [Article Motivation](https://towardsdatascience.com/speed-up-your-numpy-and-pandas-with-numexpr-package-25bd1ab0836b)
* Add [more technical analysis features](https://en.wikipedia.org/wiki/Technical_analysis).
* Wrapper to get financial data.
* Use of the Pandas multi-indexing techniques to calculate several indicators at the same time.
* Use Plotly/Streamlit to visualize features


# Credits:

Developed by Darío López Padial (aka Bukosabino) and [other contributors](https://github.com/bukosabino/ta/graphs/contributors).

Please, let me know about any comment or feedback.

Also, I am a software engineer freelance focused on Data Science using Python tools such as Pandas, Scikit-Learn, Backtrader, Zipline or Catalyst. Don't hesitate to contact me if you need something related with this library, Python, Technical Analysis, AlgoTrading, Machine Learning, etc.
