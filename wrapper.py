import pandas as pd

from volume import *
from volatility import *
from trend import *
from fundamental import *
#from momentum import *


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
    df['volume1'] = acc_dist_index(df[high], df[low], df[close], df[volume])
    df['volume2'] = on_balance_volume(df[close], df[volume])
    df['volume3'] = on_balance_volume_mean(df[close], df[volume], 10)
    df['volume5'] = chaikin_money_flow(df[high], df[low], df[close], df[volume])
    df['volume6'] = force_index(df[close], df[volume])
    df['volume7'] = ease_of_movement(df[high], df[low], df[close], df[volume], 14)
    df['volume8'] = volume_price_trend(df[close], df[volume])
    return df


def add_volatility_ta(df, high, low, close):
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['volatility1'] = average_true_range(df[high], df[low], df[close], n=14)
    df['volatility2'] = bollinger_hband(df[close], n=20, ndev=2)
    df['volatility3'] = bollinger_lband(df[close], n=20, ndev=2)
    df['volatility4'] = bollinger_mavg(df[close], n=20)
    df['volatility5'] = bollinger_hband_indicator(df[close], n=20, ndev=2)
    df['volatility6'] = bollinger_lband_indicator(df[close], n=20, ndev=2)
    df['volatility7'] = keltner_channel(df[high], df[low], df[close], n=10)
    df['volatility8'] = donchian_channel_hband(df[close], n=20)
    df['volatility9'] = donchian_channel_lband(df[close], n=20)
    df['volatility10'] = donchian_channel_hband_indicator(df[close], n=20)
    df['volatility11'] = donchian_channel_lband_indicator(df[close], n=20)    
    return df


def add_trend_ta(df, high, low, close):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['trend1'] = macd(df[close], n_fast=12, n_slow=26, n_sign=9)
    df['trend2'] = macd_signal(df[close], n_fast=12, n_slow=26, n_sign=9)
    df['trend3'] = macd_diff(df[close], n_fast=12, n_slow=26, n_sign=9)
    df['trend4'] = ema_fast(df[close], n_fast=12)
    df['trend5'] = ema_slow(df[close], n_slow=26)
    df['trend6'] = adx(df[high], df[low], df[close], n=14)
    df['trend7'] = adx_pos(df[high], df[low], df[close], n=14)
    df['trend8'] = adx_neg(df[high], df[low], df[close], n=14)
    df['trend9'] = vortex_indicator_pos(df[high], df[low], df[close], n=14)
    df['trend10'] = vortex_indicator_neg(df[high], df[low], df[close], n=14)
    df['trend11'] = abs(df['trend9'] - df['trend10'])
    df['trend12'] = trix(df[close], n=15)
    df['trend13'] = mass_index(df[high], df[low], n=9, n2=25)
    df['trend14'] = cci(df[high], df[low], df[close], n=20, c=0.015)
    df['trend15'] = dpo(df[close], n=20)
    df['trend16'] = kst(df[close], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9)
    return df


def add_momentum_ta(df, high, low, close):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    #df['momentum1'] = rsi(df[close], n=14)


def add_fa(df, close):
    """Add fundamental analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['fundamental1'] = daily_return(df[close])
    df['fundamental2'] = cumulative_return(df[close])
    return df


def add_all_ta_features(df, open, high, low, close, volume):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(df, high, low, close, volume)
    df = add_volatility_ta(df, high, low, close)
    df = add_trend_ta(df, high, low, close)
    df = add_fa(df, close)
    return df
