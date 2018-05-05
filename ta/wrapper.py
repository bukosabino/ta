# -*- coding: utf-8 -*-
import pandas as pd

from .volume import *
from .volatility import *
from .trend import *
from .momentum import *
from .others import *


def add_volume_ta(df, high, low, close, volume, fillna=False):
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['volume1'] = acc_dist_index(df[high], df[low], df[close],
                                    df[volume], fillna=fillna)
    df['volume2'] = on_balance_volume(df[close], df[volume], fillna=fillna)
    df['volume3'] = on_balance_volume_mean(df[close], df[volume], 10,
                                    fillna=fillna)
    df['volume5'] = chaikin_money_flow(df[high], df[low], df[close],
                                        df[volume], fillna=fillna)
    df['volume6'] = force_index(df[close], df[volume], fillna=fillna)
    df['volume7'] = ease_of_movement(df[high], df[low], df[close],
                                        df[volume], 14, fillna=fillna)
    df['volume8'] = volume_price_trend(df[close], df[volume], fillna=fillna)
    return df


def add_volatility_ta(df, high, low, close, fillna=False):
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['volatility1'] = average_true_range(df[high], df[low], df[close],
                                                n=14, fillna=fillna)

    df['volatility2'] = bollinger_hband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility3'] = bollinger_lband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility4'] = bollinger_mavg(df[close], n=20, fillna=fillna)
    df['volatility5'] = bollinger_hband_indicator(df[close], n=20, ndev=2,
                                                    fillna=fillna)
    df['volatility6'] = bollinger_lband_indicator(df[close], n=20, ndev=2,
                                                    fillna=fillna)

    df['volatility7'] = keltner_channel_central(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility8'] = keltner_channel_hband(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility9'] = keltner_channel_lband(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility10'] = keltner_channel_hband_indicator(df[high], df[low],
                                                df[close], n=10, fillna=fillna)
    df['volatility11'] = keltner_channel_lband_indicator(df[high], df[low],
                                                df[close], n=10, fillna=fillna)

    df['volatility12'] = donchian_channel_hband(df[close], n=20, fillna=fillna)
    df['volatility13'] = donchian_channel_lband(df[close], n=20, fillna=fillna)
    df['volatility14'] = donchian_channel_hband_indicator(df[close], n=20,
                                                            fillna=fillna)
    df['volatility15'] = donchian_channel_lband_indicator(df[close], n=20,
                                                            fillna=fillna)
    return df


def add_trend_ta(df, high, low, close, fillna=False):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['trend1'] = macd(df[close], n_fast=12, n_slow=26, fillna=fillna)
    df['trend2'] = macd_signal(df[close], n_fast=12, n_slow=26, n_sign=9,
                                    fillna=fillna)
    df['trend3'] = macd_diff(df[close], n_fast=12, n_slow=26, n_sign=9,
                                    fillna=fillna)
    df['trend4'] = ema_fast(df[close], n_fast=12, fillna=fillna)
    df['trend5'] = ema_slow(df[close], n_slow=26, fillna=fillna)
    df['trend6'] = adx(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend7'] = adx_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend8'] = adx_neg(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend9'] = adx_indicator(df[high], df[low], df[close], n=14,
                                    fillna=fillna)
    df['trend10'] = vortex_indicator_pos(df[high], df[low], df[close], n=14,
                                    fillna=fillna)
    df['trend11'] = vortex_indicator_neg(df[high], df[low], df[close], n=14,
                                    fillna=fillna)
    df['trend12'] = abs(df['trend10'] - df['trend11'])
    df['trend13'] = trix(df[close], n=15, fillna=fillna)
    df['trend14'] = mass_index(df[high], df[low], n=9, n2=25, fillna=fillna)
    df['trend15'] = cci(df[high], df[low], df[close], n=20, c=0.015,
                                    fillna=fillna)
    df['trend16'] = dpo(df[close], n=20, fillna=fillna)
    df['trend17'] = kst(df[close], r1=10, r2=15, r3=20, r4=30, n1=10,
                            n2=10, n3=10, n4=15, fillna=fillna)
    df['trend18'] = kst_sig(df[close], r1=10, r2=15, r3=20, r4=30, n1=10,
                            n2=10, n3=10, n4=15, nsig=9, fillna=fillna)
    df['trend19'] = df['trend17'] - df['trend18']
    df['trend20'] = ichimoku_a(df[high], df[low], n1=9, n2=26, fillna=fillna)
    df['trend21'] = ichimoku_b(df[high], df[low], n2=26, n3=52, fillna=fillna)
    return df


def add_momentum_ta(df, high, low, close, volume, fillna=False):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['momentum1'] = rsi(df[close], n=14, fillna=fillna)
    df['momentum2'] = money_flow_index(df[high], df[low], df[close],
                                        df[volume], n=14, fillna=fillna)
    df['momentum3'] = tsi(df[close], r=25, s=13, fillna=fillna)
    return df


def add_others_ta(df, close, fillna=False):
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['others1'] = daily_return(df[close], fillna=fillna)
    df['others2'] = cumulative_return(df[close], fillna=fillna)
    return df


def add_all_ta_features(df, open, high, low, close, volume, fillna=False):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(df, high, low, close, volume, fillna=fillna)
    df = add_volatility_ta(df, high, low, close, fillna=fillna)
    df = add_trend_ta(df, high, low, close, fillna=fillna)
    df = add_momentum_ta(df, high, low, close, volume, fillna=fillna)
    df = add_others_ta(df, close, fillna=fillna)
    return df
