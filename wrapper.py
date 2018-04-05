import pandas as pd

from volume import *
from volatility import *

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
    return df
