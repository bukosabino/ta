# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import *


def average_true_range(high, low, close, n=14):
    """Average True Range (ATR)
    https://en.wikipedia.org/wiki/Average_true_range
    """
    x = high - low
    y = abs(high - close.diff(1))
    z = abs(low - close.diff(1))
    aux = np.array([x, y, z])
    tr = pd.Series(aux.max(axis=0), index=high.index.values)
    return pd.Series(ema(tr, n).fillna(0), name='atr')


def bollinger_bands(close, n=20, ndev=2):
    """Bollinger Bands (BB)
    https://en.wikipedia.org/wiki/Bollinger_Bands
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    lband = mavg - ndev*mstd
    result = pd.Series(lband, name='lband'), pd.Series(mavg, name='mavg'),\
             pd.Series(hband, name='hband')
    return result


def bollinger_hband_indicator(close, n=20, ndev=2):
    """Bollinger High Band Indicator
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Return 1, if close is higher than bollinger hband. Else, return 0.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2):
    """Bollinger Low Band Indicator
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Return 1, if close is lower than bollinger lband. Else, return 0.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='bbilband')


def keltner_channel(high, low, close, n=10):
    """Keltner channel (KC)
    https://en.wikipedia.org/wiki/Keltner_channel
    """
    tp = (high + low + close) / 3.0
    return pd.Series(tp.rolling(n).mean(), name='kc')


def donchian_channel(close, n=20):
    """Donchian channel (DC)
    https://en.wikipedia.org/wiki/Donchian_channel
    """
    hband = close.rolling(n).max()
    lband = close.rolling(n).min()
    return pd.Series(lband, name='dclband'), pd.Series(hband, name='dchband')


def donchian_channel_hband_indicator(close, n=20):
    """Donchian High Band Indicator
    https://en.wikipedia.org/wiki/Donchian_channel
    Return 1, if close is higher than hband channel. Else, return 0.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='dcihband')


def donchian_channel_lband_indicator(close, n=20):
    """Donchian Low Band Indicator
    https://en.wikipedia.org/wiki/Donchian_channel
    Return 1, if close is lower than lband channel. Else, return 0.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='dcilband')
