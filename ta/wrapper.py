import pandas as pd

from .momentum import *
from .others import *
from .trend import * # AroonIndicator, MACD, EMAIndicator, TRIXIndicator, MassIndex
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
from .volume import *


def add_volume_ta(df, high, low, close, volume, fillna=False, colprefix=""):
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['{}volume_adi'.format(colprefix)] = acc_dist_index(df[high],
                                                          df[low],
                                                          df[close],
                                                          df[volume],
                                                          fillna=fillna)
    df['{}volume_obv'.format(colprefix)] = on_balance_volume(df[close],
                                                             df[volume],
                                                             fillna=fillna)
    df['{}volume_cmf'.format(colprefix)] = chaikin_money_flow(df[high],
                                                              df[low],
                                                              df[close],
                                                              df[volume],
                                                              fillna=fillna)
    df['{}volume_fi'.format(colprefix)] = force_index(df[close],
                                                      df[volume],
                                                      fillna=fillna)
    df['{}volume_em'.format(colprefix)] = ease_of_movement(df[high],
                                                           df[low],
                                                           df[close],
                                                           df[volume],
                                                           n=14,
                                                           fillna=fillna)
    df['{}volume_vpt'.format(colprefix)] = volume_price_trend(df[close],
                                                              df[volume],
                                                              fillna=fillna)
    df['{}volume_nvi'.format(colprefix)] = negative_volume_index(df[close],
                                                                 df[volume],
                                                                 fillna=fillna)
    return df


def add_volatility_ta(df, high, low, close, fillna=False, colprefix=""):
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Average True Range
    indicator_atr = AverageTrueRange(close=df[close], high=df[high], low=df[low], n=10, fillna=fillna)
    df[f'{colprefix}volatility_atr'] = indicator_atr.average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(close=df[close], n=20, ndev=2, fillna=fillna)
    df[f'{colprefix}volatility_bbh'] = indicator_bb.bollinger_hband()
    df[f'{colprefix}volatility_bbl'] = indicator_bb.bollinger_lband()
    df[f'{colprefix}volatility_bbm'] = indicator_bb.bollinger_mavg()
    df[f'{colprefix}volatility_bbhi'] = indicator_bb.bollinger_hband_indicator()
    df[f'{colprefix}volatility_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    indicator_kc = KeltnerChannel(close=df[close], high=df[high], low=df[low], n=10, fillna=fillna)
    df[f'{colprefix}volatility_kcc'] = indicator_kc.keltner_channel_central()
    df[f'{colprefix}volatility_kch'] = indicator_kc.keltner_channel_hband()
    df[f'{colprefix}volatility_kcl'] = indicator_kc.keltner_channel_lband()
    df[f'{colprefix}volatility_kchi'] = indicator_kc.keltner_channel_hband_indicator()
    df[f'{colprefix}volatility_kcli'] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(close=df[close], n=20, fillna=fillna)
    df[f'{colprefix}volatility_dcl'] = indicator_dc.donchian_channel_lband()
    df[f'{colprefix}volatility_dch'] = indicator_dc.donchian_channel_hband()
    df[f'{colprefix}volatility_dchi'] = indicator_dc.donchian_channel_hband_indicator()
    df[f'{colprefix}volatility_dcli'] = indicator_dc.donchian_channel_lband_indicator()

    return df


def add_trend_ta(df, high, low, close, fillna=False, colprefix=""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # MACD
    indicator_macd = MACD(close=df[close], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)
    df[f'{colprefix}trend_macd'] = indicator_macd.macd()
    df[f'{colprefix}trend_macd_signal'] = indicator_macd.macd_signal()
    df[f'{colprefix}trend_macd_diff'] = indicator_macd.macd_diff()

    # EMAs
    df[f'{colprefix}trend_ema_fast'] = EMAIndicator(
        close=df[close], n=12, fillna=fillna).ema_indicator()
    df[f'{colprefix}trend_ema_slow'] = EMAIndicator(
        close=df[close], n=26, fillna=fillna).ema_indicator()

    df['{}trend_adx'.format(colprefix)] = adx(df[high],
                                              df[low],
                                              df[close],
                                              n=14,
                                              fillna=fillna)
    df['{}trend_adx_pos'.format(colprefix)] = adx_pos(df[high],
                                                      df[low],
                                                      df[close],
                                                      n=14,
                                                      fillna=fillna)
    df['{}trend_adx_neg'.format(colprefix)] = adx_neg(df[high],
                                                      df[low],
                                                      df[close],
                                                      n=14,
                                                      fillna=fillna)
    df['{}trend_vortex_ind_pos'.format(colprefix)] = vortex_indicator_pos(
                                                                df[high],
                                                                df[low],
                                                                df[close],
                                                                n=14,
                                                                fillna=fillna)
    df['{}trend_vortex_ind_neg'.format(colprefix)] = vortex_indicator_neg(
                                                                df[high],
                                                                df[low],
                                                                df[close],
                                                                n=14,
                                                                fillna=fillna)
    df['{}trend_vortex_diff'.format(colprefix)] = abs(
        df['{}trend_vortex_ind_pos'.format(colprefix)] -
        df['{}trend_vortex_ind_neg'.format(colprefix)])

    # TRIX Indicator
    indicator = TRIXIndicator(close=df[close], n=15, fillna=fillna)
    df[f'{colprefix}trend_trix'] = indicator.trix()

    # Mass Index
    indicator = MassIndex(high=df[high], low=df[low], n=9, n2=25, fillna=fillna)
    df[f'{colprefix}trend_mass_index'] = indicator.mass_index()

    df['{}trend_cci'.format(colprefix)] = cci(df[high],
                                              df[low],
                                              df[close],
                                              n=20,
                                              c=0.015,
                                              fillna=fillna)
    df['{}trend_dpo'.format(colprefix)] = dpo(df[close], n=20, fillna=fillna)
    df['{}trend_kst'.format(colprefix)] = kst(df[close], r1=10, r2=15, r3=20,
                                              r4=30, n1=10, n2=10, n3=10,
                                              n4=15, fillna=fillna)
    df['{}trend_kst_sig'.format(colprefix)] = kst_sig(df[close], r1=10, r2=15,
                                                      r3=20, r4=30, n1=10,
                                                      n2=10, n3=10, n4=15,
                                                      nsig=9, fillna=fillna)
    df['{}trend_kst_diff'.format(colprefix)] = (
        df['{}trend_kst'.format(colprefix)] -
        df['{}trend_kst_sig'.format(colprefix)])
    df['{}trend_ichimoku_a'.format(colprefix)] = ichimoku_a(df[high], df[low],
                                                            n1=9, n2=26,
                                                            fillna=fillna)
    df['{}trend_ichimoku_b'.format(colprefix)] = ichimoku_b(df[high], df[low],
                                                            n2=26, n3=52,
                                                            fillna=fillna)
    df['{}trend_visual_ichimoku_a'.format(colprefix)] = ichimoku_a(
                                                            df[high],
                                                            df[low],
                                                            n1=9,
                                                            n2=26,
                                                            visual=True,
                                                            fillna=fillna)
    df['{}trend_visual_ichimoku_b'.format(colprefix)] = ichimoku_b(
                                                                df[high],
                                                                df[low],
                                                                n2=26,
                                                                n3=52,
                                                                visual=True,
                                                                fillna=fillna)

    # Aroon Indicator
    indicator = AroonIndicator(df[close], n=25, fillna=fillna)
    df[f'{colprefix}trend_aroon_up'] = indicator.aroon_up()
    df[f'{colprefix}trend_aroon_down'] = indicator.aroon_down()
    df[f'{colprefix}trend_aroon_ind'] = indicator.aroon_indicator()

    return df


def add_momentum_ta(df, high, low, close, volume, fillna=False, colprefix=""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['{}momentum_rsi'.format(colprefix)] = rsi(df[close], n=14,
                                                 fillna=fillna)
    df['{}momentum_mfi'.format(colprefix)] = money_flow_index(df[high],
                                                              df[low],
                                                              df[close],
                                                              df[volume],
                                                              n=14,
                                                              fillna=fillna)
    df['{}momentum_tsi'.format(colprefix)] = tsi(df[close], r=25, s=13,
                                                 fillna=fillna)
    df['{}momentum_uo'.format(colprefix)] = uo(df[high], df[low], df[close],
                                               fillna=fillna)
    df['{}momentum_stoch'.format(colprefix)] = stoch(df[high], df[low],
                                                     df[close], fillna=fillna)
    df['{}momentum_stoch_signal'.format(colprefix)] = stoch_signal(
                                                                df[high],
                                                                df[low],
                                                                df[close],
                                                                fillna=fillna)
    df['{}momentum_wr'.format(colprefix)] = wr(df[high], df[low], df[close],
                                               fillna=fillna)
    df['{}momentum_ao'.format(colprefix)] = ao(
        df[high], df[low], fillna=fillna)
    df['{}momentum_kama'.format(colprefix)] = kama(df[close], fillna=fillna)
    return df


def add_others_ta(df, close, fillna=False, colprefix=""):
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['{}others_dr'.format(colprefix)] = daily_return(df[close],
                                                       fillna=fillna)
    df['{}others_dlr'.format(colprefix)] = daily_log_return(df[close],
                                                            fillna=fillna)
    df['{}others_cr'.format(colprefix)] = cumulative_return(df[close],
                                                            fillna=fillna)
    return df


def add_all_ta_features(df, open, high, low, close, volume, fillna=False,
                        colprefix=""):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(df, high, low, close, volume, fillna=fillna,
                       colprefix=colprefix)
    df = add_volatility_ta(df, high, low, close, fillna=fillna,
                           colprefix=colprefix)
    df = add_trend_ta(df, high, low, close, fillna=fillna, colprefix=colprefix)
    df = add_momentum_ta(df, high, low, close, volume, fillna=fillna,
                         colprefix=colprefix)
    df = add_others_ta(df, close, fillna=fillna, colprefix=colprefix)
    return df
