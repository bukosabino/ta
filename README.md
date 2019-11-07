# Technical Analysis Library in Python

[![CircleCI](https://circleci.com/gh/bukosabino/ta/tree/master.svg?style=svg)](https://circleci.com/gh/bukosabino/ta/tree/master)

It is a Technical Analysis library to financial time series datasets (open, close, high, low, volume). You can use it to do feature engineering from financial datasets. It is builded on Python Pandas library.

![alt text](https://raw.githubusercontent.com/bukosabino/ta/master/doc/figure.png)

The library has implemented 32 indicators:

#### Volume

* Accumulation/Distribution Index (ADI)
* On-Balance Volume (OBV)
* Chaikin Money Flow (CMF)
* Force Index (FI)
* Ease of Movement (EoM, EMV)
* Volume-price Trend (VPT)
* Negative Volume Index (NVI)

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

* English: https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543
* Spanish: https://medium.com/datos-y-ciencia/biblioteca-de-an%C3%A1lisis-t%C3%A9cnico-sobre-series-temporales-financieras-para-machine-learning-con-cb28f9427d0


# How to use (python >= 3.6)

```sh
$ pip install --upgrade ta
```

To use this library you should have a financial time series dataset including “Timestamp”, “Open”, “High”, “Low”, “Close” and “Volume” columns.

You should clean or fill NaN values in your dataset before add technical analysis features.

You can get code examples in [examples_to_use](https://github.com/bukosabino/ta/tree/master/examples_to_use) folder.

You can visualize the features in [this notebook](https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb).


#### Example adding all features

```python
import pandas as pd
import ta

# Load datas
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = ta.utils.dropna(df)

# Add ta features filling NaN values
df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=True)
```


#### Example adding individual features

```python
import pandas as pd
import ta

# Load datas
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = ta.utils.dropna(df)

# Add bollinger band high indicator filling NaN values
df['bb_high_indicator'] = ta.bollinger_hband_indicator(df["Close"], n=20, ndev=2, fillna=True)

# Add bollinger band low indicator filling NaN values
df['bb_low_indicator'] = ta.bollinger_lband_indicator(df["Close"], n=20, ndev=2, fillna=True)
```


# Deploy and develop (for developers)

```sh
$ git clone https://github.com/bukosabino/ta.git
$ cd ta
$ pip install -r requirements.txt
$ make lint
$ make test
```


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://pandas.pydata.org
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# In Progress:

* automated tests for indicators.
* coverage
* CI repo


# TODO:

* add [more technical analysis features](https://en.wikipedia.org/wiki/Technical_analysis).
* use Dask library to parallelize the implementation
* use Dash/Streamlit to visualize features


# Credits:

Developed by Darío López Padial (aka Bukosabino) and other contributors: https://github.com/bukosabino/ta/graphs/contributors

Please, let me know about any comment or feedback.

Also, I am a software freelance focused on Data Science using Python tools such as Pandas, Scikit-Learn, Backtrader, Zipline or Catalyst. Don't hesitate to contact me if you need something related with this library, Python, Technical Analysis, AlgoTrading, Machine Learning, etc.
