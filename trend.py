# -*- coding: utf-8 -*-
import pandas as pd
from pandas.stats import moments
import numpy as np

from utils import *


def macd(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD)
    https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = pd.Series(emafast - emaslow, name='MACD_%d_%d' % (n_fast, n_slow))
    return pd.Series(macd, name='MACD_%d_%d' % (n_fast, n_slow))


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Signal)
    https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = pd.Series(emafast - emaslow, name='MACD_%d_%d' % (n_fast, n_slow))
    return pd.Series(ema(macd, n_sign), name='MACD_sign_%d_%d' % (n_fast, n_slow))

    
def macd_diff(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Diff)
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = pd.Series(emafast - emaslow, name='MACD_%d_%d' % (n_fast, n_slow))
    macdsign = pd.Series(ema(macd, n_sign), name='MACD_sign_%d_%d' % (n_fast, n_slow))
    return pd.Series(macd - macdsign, name='MACD_diff_%d_%d' % (n_fast, n_slow))


def ema_fast(close, n_fast=12):
    emafast = ema(close, n_fast)
    return pd.Series(emafast, name='emafast')


def ema_slow(close, n_slow=26):
    emaslow = ema(close, n_slow)
    return pd.Series(emaslow, name='emaslow')



def adx(high, low, close, n=14):
    """Average Directional Movement Index (ADX)
    """
    cs = close.shift(1)
    
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()

    up = high - high.shift(1)
    dn = low.shift(1) - low

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    
    dip = 100 * pos.rolling(n).sum() / trs
    din = 100 * neg.rolling(n).sum() / trs

    dx = 100 * np.abs((dip - din)/(dip + din))
    adx = dx.ewm(n).mean()
    return pd.Series(adx, name='adx')


def adx_pos(high, low, close, n=14):
    """Average Directional Movement Index Positive (ADX)
    """
    cs = close.shift(1)
    
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()

    up = high - high.shift(1)
    dn = low.shift(1) - low

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    
    dip = 100 * pos.rolling(n).sum() / trs
    return pd.Series(dip, name='adx_pos')


def adx_neg(high, low, close, n=14):
    """Average Directional Movement Index Negative (ADX)
    """
    cs = close.shift(1)
    
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()

    up = high - high.shift(1)
    dn = low.shift(1) - low

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    
    din = 100 * neg.rolling(n).sum() / trs
    return pd.Series(din, name='adx_neg')


def vortex_indicator_pos(high, low, close, n=14):
    """Vortex Indicator (VI)
    """    
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vip = vmp.rolling(n).sum() / trn
    return pd.Series(vip, name='vip')


def vortex_indicator_neg(high, low, close, n=14):
    """Vortex Indicator (VI)
    """    
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    return pd.Series(vin, name='vin')


def trix(close, n=15):
    """ Trix (TRIX)
    """
    ema1 = close.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    ema3 = ema2.ewm(span=n, min_periods=n-1).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
    return pd.Series(trix*100, name='trix_'+str(n))
