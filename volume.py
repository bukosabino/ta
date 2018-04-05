import pandas as pd


def add_volume_ta(df, high, low, close, volume):
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['v1'] = acc_dist_index(df[high], df[low], df[close], df[volume])
    df['v2'] = on_balance_volume(df[close], df[volume])
    df['v3'] = on_balance_volume_mean(df[close], df[volume], 10)
    df['v5'] = chaikin_money_flow(df[high], df[low], df[close], df[volume])
    df['v6'] = force_index(df[close], df[volume])
    df['v7'] = ease_of_movement(df[high], df[low], df[close], df[volume], 14)
    df['v8'] = volume_price_trend(df[close], df[volume])
    return df


def acc_dist_index(high, low, close, volume):
    """Accumulation/Distribution Index (ADI)
    https://en.wikipedia.org/wiki/Accumulation/distribution_index
    Acting as leading indicator of price movements.
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0) # float division by zero
    ad = clv * volume
    ad = ad + ad.shift(1)
    return pd.Series(ad, name='adi')


def on_balance_volume(close, volume):
    """On-balance volume (OBV)
    https://en.wikipedia.org/wiki/On-balance_volume
    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return df['OBV']


def on_balance_volume_mean(close, volume, n=10):
    """On-balance volume mean (OBV mean)
    https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    It's based on a cumulative total volume.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return pd.Series(df['OBV'].rolling(n).mean(), name='obv')


def chaikin_money_flow(high, low, close, volume, n=20):
    """Chaikin Money Flow (CMF)
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    It measures the amount of Money Flow Volume over a specific period.
    """
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0) # float division by zero
    mfv *= volume
    return pd.Series(mfv.rolling(n).sum() / volume.rolling(n).sum(), name='cmf')


def force_index(close, volume, n=2):
    """Force Index (FI)
    https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.
    """
    return pd.Series(close.diff(n) * volume.diff(n), name='fi_'+str(n))


def ease_of_movement(high, low, close, volume, n=20):
    """Ease of movement (EoM, EMV)
    https://en.wikipedia.org/wiki/Ease_of_movement
    https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.
    """
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    return pd.Series(emv.rolling(n).mean(), name='eom_' + str(n))


def volume_price_trend(close, volume):
    # TODO: problems with g-research challenge datas (too big values)
    """Volume-price trend (VPT)
    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend
    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.
    """
    vpt = volume * ((close - close.shift(1)) / close.shift(1))
    vpt = vpt.shift(1) + vpt
    return pd.Series(vpt, name='vpt')


# TODO

def negative_volume_index():
    """Negative Volume Index (NVI)
    https://en.wikipedia.org/wiki/Negative_volume_index
    """
    # TODO
    return


def put_call_ratio():
    """Put/Call ratio (PCR)
    https://en.wikipedia.org/wiki/Put/call_ratio
    """
    # TODO
    return
