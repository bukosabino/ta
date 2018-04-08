import pandas as pd
import numpy as np


def macd(close, n_fast=12, n_slow=26):
    """Moving Average Convergence Divergence (MACD)
    
    https://en.wikipedia.org/wiki/MACD
    
    Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    return pd.Series(macd, name='MACD_%d_%d' % (n_fast, n_slow))


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Signal)
    
    https://en.wikipedia.org/wiki/MACD

    Shows EMA of MACD.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    return pd.Series(macd.ewm(n_sign).mean(),
                        name='MACD_sign_%d_%d' % (n_fast, n_slow))


def macd_diff(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Diff)
    
    https://en.wikipedia.org/wiki/MACD
        
    Shows the relationship between MACD and MACD Signal.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    macdsign = macd.ewm(n_sign).mean()
    return pd.Series(macd - macdsign, name='MACD_diff_%d_%d' % (n_fast, n_slow))


def ema_fast(close, n_fast=12):
    """EMA
    
    Short Period Exponential Moving Average
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = close.ewm(n_fast).mean()
    return pd.Series(emafast, name='emafast')


def ema_slow(close, n_slow=26):
    """EMA

    Long Period Exponential Moving Average
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_slow(int): n period long-term.

    Returns:
        pandas.Series: New feature generated.
    """
    emaslow = close.ewm(n_slow).mean()
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
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    
    Shows the percent rate of change of a triple exponentially smoothed moving average.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    
    Returns:
        pandas.Series: New feature generated.
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
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    
    Is an indicator designed to remove trend from price and make it easier to identify cycles.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    
    Returns:
        pandas.Series: New feature generated.
    """
    dpo = close.shift(int(n/(2+1))) - close.rolling(n).mean()
    return pd.Series(dpo, name='dpo_'+str(n))


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    """KST Oscillator (KST)
    """
    rocma1 = (close / close.shift(r1) - 1).rolling(n1).mean()
    rocma2 = (close / close.shift(r2) - 1).rolling(n2).mean()
    rocma3 = (close / close.shift(r3) - 1).rolling(n3).mean()
    rocma4 = (close / close.shift(r4) - 1).rolling(n4).mean()
    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)
    sig = kst.rolling(nsig).mean()
    return pd.Series(sig, name='sig')


def ichimoku_a(high, low, n1=9, n2=26, n3=52):
    """Ichimoku Kinkō Hyō (Ichimoku)
    """
    conv = (high.rolling(n1).max() + low.rolling(n1).min()) / 2
    base = (high.rolling(n2).max() + low.rolling(n2).min()) / 2

    spana = (conv + base) / 2
    spanb = (high.rolling(n3).max() + low.rolling(n3).min()) / 2

    return pd.Series(spana.shift(n2), name='ichimoku_'+str(n2))


def ichimoku_b(high, low, n1=9, n2=26, n3=52):
    """Ichimoku Kinkō Hyō (Ichimoku)
    """
    conv = (high.rolling(n1).max() + low.rolling(n1).min()) / 2
    base = (high.rolling(n2).max() + low.rolling(n2).min()) / 2

    spana = (conv + base) / 2
    spanb = (high.rolling(n3).max() + low.rolling(n3).min()) / 2

    return pd.Series(spanb.shift(n2), name='ichimoku_'+str(n2))
