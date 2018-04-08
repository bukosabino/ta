import pandas as pd
import numpy as np


def average_true_range(high, low, close, n=14):
    """Average True Range (ATR)
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    
    The indicator provide an indication of the degree of price volatility. Strong moves, in either direction, are often accompanied by large ranges, or large True Ranges.
    
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    return pd.Series(tr.ewm(n).mean(), name='atr')


def bollinger_mavg(close, n=20):
    """Bollinger Bands (BB)
    
    https://en.wikipedia.org/wiki/Bollinger_Bands
    
    N-period simple moving average (MA).
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    return pd.Series(mavg, name='mavg')


def bollinger_hband(close, n=20, ndev=2):
    """Bollinger Bands (BB)
    
    https://en.wikipedia.org/wiki/Bollinger_Bands
    
    Upper band at K times an N-period standard deviation above the moving average (MA + Kσ).
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    return pd.Series(hband, name='hband')


def bollinger_lband(close, n=20, ndev=2):
    """Bollinger Bands (BB)
    
    https://en.wikipedia.org/wiki/Bollinger_Bands
    
    Lower band at K times an N-period standard deviation below the moving average (MA − Kσ).
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    return pd.Series(lband, name='lband')


def bollinger_hband_indicator(close, n=20, ndev=2):
    """Bollinger High Band Indicator
    
    https://en.wikipedia.org/wiki/Bollinger_Bands
    
    Return 1, if close is higher than bollinger high band. Else, return 0.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        
    Returns:
        pandas.Series: New feature generated.
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
    
    Return 1, if close is lower than bollinger low band. Else, return 0.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='bbilband')


def keltner_channel_central(high, low, close, n=10):
    """Keltner channel (KC)
    
    https://en.wikipedia.org/wiki/Keltner_channel
    
    Showing a simple moving average line (central) of typical price.
    
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = (high + low + close) / 3.0
    return pd.Series(tp.rolling(n).mean(), name='kc_central')


def keltner_channel_hband(high, low, close, n=10):
    """Keltner channel (KC)
    
    https://en.wikipedia.org/wiki/Keltner_channel
    
    Showing a simple moving average line (high) of typical price.
    
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((4*high) - (2*low) + close) / 3.0
    return pd.Series(tp.rolling(n).mean(), name='kc_hband')


def keltner_channel_lband(high, low, close, n=10):
    """Keltner channel (KC)
    
    https://en.wikipedia.org/wiki/Keltner_channel
    
    Showing a simple moving average line (low) of typical price.
    
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((-2*high) + (4*low) + close) / 3.0
    return pd.Series(tp.rolling(n).mean(), name='kc_lband')
    
    
def keltner_channel_hband_indicator(high, low, close, n=10):
    """Keltner Channel High Band Indicator (KC)

    https://en.wikipedia.org/wiki/Keltner_channel
    
    Return 1, if close is higher than keltner high band channel. Else, return 0.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = ((4*high) - (2*low) + close) / 3.0
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='kci_hband')


def keltner_channel_lband_indicator(high, low, close, n=10):
    """Keltner Channel Low Band Indicator (KC)

    https://en.wikipedia.org/wiki/Keltner_channel
    
    Return 1, if close is lower than keltner low band channel. Else, return 0.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = ((-2*high) + (4*low) + close) / 3.0
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='kci_lband')


def donchian_channel_hband(close, n=20):
    """Donchian channel (DC)
    
    https://www.investopedia.com/terms/d/donchianchannels.asp
    
    The upper band marks the highest price of an issue for n periods.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        
    Returns:
        pandas.Series: New feature generated.
    """
    hband = close.rolling(n).max()
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(close, n=20):
    """Donchian channel (DC)
    
    https://www.investopedia.com/terms/d/donchianchannels.asp
    
    The lower band marks the lowest price for n periods.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
                
    Returns:
        pandas.Series: New feature generated.
    """
    lband = close.rolling(n).min()
    return pd.Series(lband, name='dclband')


def donchian_channel_hband_indicator(close, n=20):
    """Donchian High Band Indicator
    
    https://www.investopedia.com/terms/d/donchianchannels.asp
    
    Return 1, if close is higher than donchian high band channel. Else, return 0.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close >= hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='dcihband')


def donchian_channel_lband_indicator(close, n=20):
    """Donchian Low Band Indicator

    https://www.investopedia.com/terms/d/donchianchannels.asp
    
    Return 1, if close is lower than donchian low band channel. Else, return 0.

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close <= lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='dcilband')
