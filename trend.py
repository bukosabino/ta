import pandas as pd
#from pandas.stats import moments
import numpy as np

from utils import *


def macd(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD)
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = pd.Series(emafast - emaslow, name='MACD_%d_%d' % (n_fast, n_slow))
    return pd.Series(macd, name='MACD_%d_%d' % (n_fast, n_slow))


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Signal)
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
    """Trix (TRIX)
    """
    ema1 = close.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    ema3 = ema2.ewm(span=n, min_periods=n-1).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
    return pd.Series(trix*100, name='trix_'+str(n))


def mass_index(high, low, n=9, n2=25):
    """Mass Index (MI)
    """
    amplitude = high - low
    ema1 = amplitude.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    mass = ema1/ema2
    return pd.Series(mass.rolling(n2).sum(), name='mass_index_'+str(n))


def cci(high, low, close, n=20, c=0.015):
    """Commodity Channel Index (CCI)
    """
    pp = (high+low+close)/3
    cci = (pp-pp.rolling(n).mean())/pp.rolling(n).std()
    return pd.Series(1/c * cci, name='cci')


def dpo(close, n=20):
    """Detrended Price Oscillator (DPO)
    """
    dpo = close.shift(int(n/(2+1))) - close.rolling(n).mean()
    return pd.Series(dpo, name='dpo_'+str(n))
    
def kst():
    
    
    ROC1 = close.diff(r1 - 1) / close.shift(r1 - 1)
    
    rocma1 = moments.rolling_mean(close / close.shift(r1) - 1, n1)


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    """KST Oscillator (KST)
    """
    rocma1 = (close / close.shift(r1) - 1).rolling(n1).mean()
    rocma2 = (close / close.shift(r2) - 1).rolling(n2).mean()
    rocma3 = (close / close.shift(r3) - 1).rolling(n3).mean()
    rocma4 = (close / close.shift(r4) - 1).rolling(n4).mean()
    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)    
    sig = kst.rolling(nsig).mean()
    #return pd.Series(kst, name='kst')
    return pd.Series(sig, name='sig')