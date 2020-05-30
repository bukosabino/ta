"""
.. module:: wrapper
   :synopsis: Wrapper of Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)
"""

import pandas as pd

from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,
                         ROCIndicator, RSIIndicator, StochasticOscillator,
                         TSIIndicator, UltimateOscillator, WilliamsRIndicator)
from ta.others import (CumulativeReturnIndicator, DailyLogReturnIndicator,
                       DailyReturnIndicator)
from ta.trend import (MACD, ADXIndicator, AroonIndicator, CCIIndicator,
                      DPOIndicator, EMAIndicator, IchimokuIndicator,
                      KSTIndicator, MassIndex, PSARIndicator, SMAIndicator,
                      TRIXIndicator, VortexIndicator)
from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           KeltnerChannel)
from ta.volume import (AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
                       EaseOfMovementIndicator, ForceIndexIndicator,
                       MFIIndicator, NegativeVolumeIndexIndicator,
                       OnBalanceVolumeIndicator, VolumePriceTrendIndicator,
                       VolumeWeightedAveragePrice)


def add_volume_ta(df: pd.DataFrame, high: str, low: str, close: str, volume: str,
                  fillna: bool = False, colprefix: str = "") -> pd.DataFrame:
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

    # Accumulation Distribution Index
    df[f'{colprefix}volume_adi'] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna).acc_dist_index()

    # On Balance Volume
    df[f'{colprefix}volume_obv'] = OnBalanceVolumeIndicator(
        close=df[close], volume=df[volume], fillna=fillna).on_balance_volume()

    # Chaikin Money Flow
    df[f'{colprefix}volume_cmf'] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna).chaikin_money_flow()

    # Force Index
    df[f'{colprefix}volume_fi'] = ForceIndexIndicator(
        close=df[close], volume=df[volume], n=13, fillna=fillna).force_index()

    # Money Flow Indicator
    df[f'{colprefix}momentum_mfi'] = MFIIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], n=14, fillna=fillna).money_flow_index()

    # Ease of Movement
    indicator = EaseOfMovementIndicator(high=df[high], low=df[low], volume=df[volume], n=14, fillna=fillna)
    df[f'{colprefix}volume_em'] = indicator.ease_of_movement()
    df[f'{colprefix}volume_sma_em'] = indicator.sma_ease_of_movement()

    # Volume Price Trend
    df[f'{colprefix}volume_vpt'] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna).volume_price_trend()

    # Negative Volume Index
    df[f'{colprefix}volume_nvi'] = NegativeVolumeIndexIndicator(
        close=df[close], volume=df[volume], fillna=fillna).negative_volume_index()

    # Volume Weighted Average Price
    df[f'{colprefix}volume_vwap'] = VolumeWeightedAveragePrice(
        high=df[high], low=df[low], close=df[close], volume=df[volume], n=14, fillna=fillna
    ).volume_weighted_average_price()

    return df


def add_volatility_ta(df: pd.DataFrame, high: str, low: str, close: str,
                      fillna: bool = False, colprefix: str = "") -> pd.DataFrame:
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
    df[f'{colprefix}volatility_atr'] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], n=10, fillna=fillna).average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(close=df[close], n=20, ndev=2, fillna=fillna)
    df[f'{colprefix}volatility_bbm'] = indicator_bb.bollinger_mavg()
    df[f'{colprefix}volatility_bbh'] = indicator_bb.bollinger_hband()
    df[f'{colprefix}volatility_bbl'] = indicator_bb.bollinger_lband()
    df[f'{colprefix}volatility_bbw'] = indicator_bb.bollinger_wband()
    df[f'{colprefix}volatility_bbp'] = indicator_bb.bollinger_pband()
    df[f'{colprefix}volatility_bbhi'] = indicator_bb.bollinger_hband_indicator()
    df[f'{colprefix}volatility_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    indicator_kc = KeltnerChannel(close=df[close], high=df[high], low=df[low], n=10, fillna=fillna)
    df[f'{colprefix}volatility_kcc'] = indicator_kc.keltner_channel_mband()
    df[f'{colprefix}volatility_kch'] = indicator_kc.keltner_channel_hband()
    df[f'{colprefix}volatility_kcl'] = indicator_kc.keltner_channel_lband()
    df[f'{colprefix}volatility_kcw'] = indicator_kc.keltner_channel_wband()
    df[f'{colprefix}volatility_kcp'] = indicator_kc.keltner_channel_pband()
    df[f'{colprefix}volatility_kchi'] = indicator_kc.keltner_channel_hband_indicator()
    df[f'{colprefix}volatility_kcli'] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(high=df[high], low=df[low], close=df[close], n=20, offset=0, fillna=fillna)
    df[f'{colprefix}volatility_dcl'] = indicator_dc.donchian_channel_lband()
    df[f'{colprefix}volatility_dch'] = indicator_dc.donchian_channel_hband()
    df[f'{colprefix}volatility_dcm'] = indicator_dc.donchian_channel_mband()
    df[f'{colprefix}volatility_dcw'] = indicator_dc.donchian_channel_wband()
    df[f'{colprefix}volatility_dcp'] = indicator_dc.donchian_channel_pband()

    return df


def add_trend_ta(df: pd.DataFrame, high: str, low: str, close: str, fillna: bool = False,
                 colprefix: str = "") -> pd.DataFrame:
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
    indicator_macd = MACD(close=df[close], n_slow=26, n_fast=12, n_sign=9, fillna=fillna)
    df[f'{colprefix}trend_macd'] = indicator_macd.macd()
    df[f'{colprefix}trend_macd_signal'] = indicator_macd.macd_signal()
    df[f'{colprefix}trend_macd_diff'] = indicator_macd.macd_diff()

    # SMAs
    df[f'{colprefix}trend_sma_fast'] = SMAIndicator(
        close=df[close], n=12, fillna=fillna).sma_indicator()
    df[f'{colprefix}trend_sma_slow'] = SMAIndicator(
        close=df[close], n=26, fillna=fillna).sma_indicator()

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
    indicator = VortexIndicator(high=df[high], low=df[low], close=df[close], n=14, fillna=fillna)
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
    df[f'{colprefix}trend_ichimoku_conv'] = indicator.ichimoku_conversion_line()
    df[f'{colprefix}trend_ichimoku_base'] = indicator.ichimoku_base_line()
    df[f'{colprefix}trend_ichimoku_a'] = indicator.ichimoku_a()
    df[f'{colprefix}trend_ichimoku_b'] = indicator.ichimoku_b()
    indicator = IchimokuIndicator(high=df[high], low=df[low], n1=9, n2=26, n3=52, visual=True, fillna=fillna)
    df[f'{colprefix}trend_visual_ichimoku_a'] = indicator.ichimoku_a()
    df[f'{colprefix}trend_visual_ichimoku_b'] = indicator.ichimoku_b()

    # Aroon Indicator
    indicator = AroonIndicator(close=df[close], n=25, fillna=fillna)
    df[f'{colprefix}trend_aroon_up'] = indicator.aroon_up()
    df[f'{colprefix}trend_aroon_down'] = indicator.aroon_down()
    df[f'{colprefix}trend_aroon_ind'] = indicator.aroon_indicator()

    # PSAR Indicator
    indicator = PSARIndicator(high=df[high], low=df[low], close=df[close], step=0.02, max_step=0.20, fillna=fillna)
    # df[f'{colprefix}trend_psar'] = indicator.psar()
    df[f'{colprefix}trend_psar_up'] = indicator.psar_up()
    df[f'{colprefix}trend_psar_down'] = indicator.psar_down()
    df[f'{colprefix}trend_psar_up_indicator'] = indicator.psar_up_indicator()
    df[f'{colprefix}trend_psar_down_indicator'] = indicator.psar_down_indicator()

    return df


def add_momentum_ta(df: pd.DataFrame, high: str, low: str, close: str, volume: str,
                    fillna: bool = False, colprefix: str = "") -> pd.DataFrame:
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
    df[f'{colprefix}momentum_rsi'] = RSIIndicator(close=df[close], n=14, fillna=fillna).rsi()

    # TSI Indicator
    df[f'{colprefix}momentum_tsi'] = TSIIndicator(close=df[close], r=25, s=13, fillna=fillna).tsi()

    # Ultimate Oscillator
    df[f'{colprefix}momentum_uo'] = UltimateOscillator(
        high=df[high], low=df[low], close=df[close], s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
        fillna=fillna).uo()

    # Stoch Indicator
    indicator = StochasticOscillator(high=df[high], low=df[low], close=df[close], n=14, d_n=3, fillna=fillna)
    df[f'{colprefix}momentum_stoch'] = indicator.stoch()
    df[f'{colprefix}momentum_stoch_signal'] = indicator.stoch_signal()

    # Williams R Indicator
    df[f'{colprefix}momentum_wr'] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna).wr()

    # Awesome Oscillator
    df[f'{colprefix}momentum_ao'] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], s=5, len=34, fillna=fillna).ao()

    # KAMA
    df[f'{colprefix}momentum_kama'] = KAMAIndicator(
        close=df[close], n=10, pow1=2, pow2=30, fillna=fillna).kama()

    # Rate Of Change
    df[f'{colprefix}momentum_roc'] = ROCIndicator(close=df[close], n=12, fillna=fillna).roc()
    return df


def add_others_ta(df: pd.DataFrame, close: str, fillna: bool = False, colprefix: str = "") -> pd.DataFrame:
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    # Daily Return
    df[f'{colprefix}others_dr'] = DailyReturnIndicator(close=df[close], fillna=fillna).daily_return()

    # Daily Log Return
    df[f'{colprefix}others_dlr'] = DailyLogReturnIndicator(close=df[close], fillna=fillna).daily_log_return()

    # Cumulative Return
    df[f'{colprefix}others_cr'] = CumulativeReturnIndicator(
        close=df[close], fillna=fillna).cumulative_return()

    return df


def add_all_ta_features(df: pd.DataFrame, open: str, high: str, low: str,
                        close: str, volume: str, fillna: bool = False, colprefix: str = "") -> pd.DataFrame:
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
    df = add_volume_ta(df=df, high=high, low=low, close=close, volume=volume, fillna=fillna, colprefix=colprefix)
    df = add_volatility_ta(df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix)
    df = add_trend_ta(df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix)
    df = add_momentum_ta(df=df, high=high, low=low, close=close, volume=volume, fillna=fillna, colprefix=colprefix)
    df = add_others_ta(df=df, close=close, fillna=fillna, colprefix=colprefix)
    return df
