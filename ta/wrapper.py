import pandas as pd

from ta.momentum import *  # RSIIndicator, MFIIndicator, TSIIndicator, UltimateOscillatorIndicator, StochIndicator
from ta.others import *
from ta.trend import (MACD, ADXIndicator, AroonIndicator, CCIIndicator,
                      DPOIndicator, EMAIndicator, IchimokuIndicator,
                      KSTIndicator, MassIndex, TRIXIndicator, VortexIndicator)
from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           KeltnerChannel)
from ta.volume import *


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
    df[f'{colprefix}volume_adi'] = acc_dist_index(
        df[high], df[low], df[close], df[volume], fillna=fillna)
    df[f'{colprefix}volume_obv'] = on_balance_volume(
        df[close], df[volume], fillna=fillna)
    df[f'{colprefix}volume_cmf'] = chaikin_money_flow(
        df[high], df[low], df[close], df[volume], fillna=fillna)
    df[f'{colprefix}volume_fi'] = force_index(
        df[close], df[volume], fillna=fillna)
    df[f'{colprefix}volume_em'] = ease_of_movement(
        df[high], df[low], df[close], df[volume], n=14, fillna=fillna)
    df[f'{colprefix}volume_vpt'] = volume_price_trend(
        df[close], df[volume], fillna=fillna)
    df[f'{colprefix}volume_nvi'] = negative_volume_index(
        df[close], df[volume], fillna=fillna)
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

    # Average Directional Movement Index (ADX)
    indicator = ADXIndicator(high=df[high], low=df[low], close=df[close], n=14, fillna=fillna)
    df[f'{colprefix}trend_adx'] = indicator.adx()
    df[f'{colprefix}trend_adx_pos'] = indicator.adx_pos()
    df[f'{colprefix}trend_adx_neg'] = indicator.adx_neg()

    # Vortex Indicator
    indicator = VortexIndicator(high=df[high], low=df[low], close=df[close], n=14, fillna=False)
    df[f'{colprefix}trend_vortex_ind_pos'] = indicator.vortex_indicator_pos()
    df[f'{colprefix}trend_vortex_ind_neg'] = indicator.vortex_indicator_neg()
    df[f'{colprefix}trend_vortex_ind_diff'] = indicator.vortex_indicator_diff()

    # TRIX Indicator
    indicator = TRIXIndicator(close=df[close], n=15, fillna=fillna)
    df[f'{colprefix}trend_trix'] = indicator.trix()

    # Mass Index
    indicator = MassIndex(high=df[high], low=df[low], n=9, n2=25, fillna=fillna)
    df[f'{colprefix}trend_mass_index'] = indicator.mass_index()

    # CCI Indicator
    indicator = CCIIndicator(high=df[high], low=df[low], close=df[close], n=20, c=0.015, fillna=fillna)
    df[f'{colprefix}trend_cci'] = indicator.cci()

    # DPO Indicator
    indicator = DPOIndicator(close=df[close], n=20, fillna=fillna)
    df[f'{colprefix}trend_dpo'] = indicator.dpo()

    # KST Indicator
    indicator = KSTIndicator(close=df[close],
                             r1=10, r2=15, r3=20,
                             r4=30, n1=10, n2=10, n3=10,
                             n4=15, nsig=9, fillna=fillna)
    df[f'{colprefix}trend_kst'] = indicator.kst()
    df[f'{colprefix}trend_kst_sig'] = indicator.kst_sig()
    df[f'{colprefix}trend_kst_diff'] = indicator.kst_diff()

    # Ichimoku Indicator
    indicator = IchimokuIndicator(high=df[high], low=df[low], n1=9, n2=26, n3=52, visual=False, fillna=fillna)
    df[f'{colprefix}trend_ichimoku_a'] = indicator.ichimoku_a()
    df[f'{colprefix}trend_ichimoku_b'] = indicator.ichimoku_b()
    indicator = IchimokuIndicator(high=df[high], low=df[low], n1=9, n2=26, n3=52, visual=True, fillna=fillna)
    df[f'{colprefix}trend_visual_ichimoku_a'] = indicator.ichimoku_a()
    df[f'{colprefix}trend_visual_ichimoku_b'] = indicator.ichimoku_b()

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

    # Relative Strength Index (RSI)
    indicator = RSIIndicator(close=df[close], n=14, fillna=fillna)
    df[f'{colprefix}momentum_rsi'] = indicator.rsi()

    # Money Flow Indicator
    indicator = MFIIndicator(high=df[high], low=df[low], close=df[close], volume=df[volume], n=14, fillna=fillna)
    df[f'{colprefix}momentum_mfi'] = indicator.money_flow_index()

    # TSI Indicator
    indicator = TSIIndicator(close=df[close], r=25, s=13, fillna=fillna)
    df[f'{colprefix}momentum_tsi'] = indicator.tsi()

    # Ultimate Oscillator
    df[f'{colprefix}momentum_uo'] = UltimateOscillatorIndicator(
        high=df[high], low=df[low], close=df[close], s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
        fillna=fillna).uo()

    # Stoch Indicator
    indicator = StochIndicator(high=df[high], low=df[low], close=df[close], n=14, d_n=3, fillna=fillna)
    df[f'{colprefix}momentum_stoch'] = indicator.stoch()
    df[f'{colprefix}momentum_stoch_signal'] = indicator.stoch_signal()

    df[f'{colprefix}momentum_wr'] = wr(
        df[high], df[low], df[close], fillna=fillna)
    df[f'{colprefix}momentum_ao'] = ao(
        df[high], df[low], fillna=fillna)

    # KAMA
    df[f'{colprefix}momentum_kama'] = KAMAIndicator(close=df[close], n=10, pow1=2, pow2=30, fillna=fillna).kama()

    # Rate Of Change
    df[f'{colprefix}momentum_roc'] = ROCIndicator(close=df[close], n=12, fillna=fillna).roc()
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
    df[f'{colprefix}others_dr'] = daily_return(df[close], fillna=fillna)
    df[f'{colprefix}others_dlr'] = daily_log_return(df[close], fillna=fillna)
    df[f'{colprefix}others_cr'] = cumulative_return(df[close], fillna=fillna)
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
    """
    df = add_volume_ta(df, high, low, close, volume, fillna=fillna, colprefix=colprefix)
    df = add_volatility_ta(df, high, low, close, fillna=fillna, colprefix=colprefix)
    df = add_trend_ta(df, high, low, close, fillna=fillna, colprefix=colprefix)
    """
    df = add_momentum_ta(df, high, low, close, volume, fillna=fillna, colprefix=colprefix)
    """
    df = add_others_ta(df, close, fillna=fillna, colprefix=colprefix)
    """
    return df
