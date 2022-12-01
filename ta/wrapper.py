"""
.. module:: wrapper
   :synopsis: Wrapper of Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)
"""
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Union

from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


def add_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Accumulation Distribution Index
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()

    # On Balance Volume
    df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).on_balance_volume()

    # Chaikin Money Flow
    df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    df[f"{colprefix}volume_fi"] = ForceIndexIndicator(
        close=df[close], volume=df[volume], window=13, fillna=fillna
    ).force_index()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(
        high=df[high], low=df[low], volume=df[volume], window=14, fillna=fillna
    )
    df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
    df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()

    # Volume Weighted Average Price
    df[f"{colprefix}volume_vwap"] = VolumeWeightedAveragePrice(
        high=df[high],
        low=df[low],
        close=df[close],
        volume=df[volume],
        window=14,
        fillna=fillna,
    ).volume_weighted_average_price()

    if not vectorized:
        # Money Flow Indicator
        df[f"{colprefix}volume_mfi"] = MFIIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            window=14,
            fillna=fillna,
        ).money_flow_index()

        # Negative Volume Index
        df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
            close=df[close], volume=df[volume], fillna=fillna
        ).negative_volume_index()

    return df


def add_volatility_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], window=20, window_dev=2, fillna=fillna
    )
    df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
    df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
    df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
    df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
    df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
    df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
    df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    indicator_kc = KeltnerChannel(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    )
    df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
    df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
    df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
    df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
    df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
    df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
    df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], window=20, offset=0, fillna=fillna
    )
    df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
    df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
    df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
    df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
    df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    if not vectorized:
        # Average True Range
        df[f"{colprefix}volatility_atr"] = AverageTrueRange(
            close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
        ).average_true_range()

        # Ulcer Index
        df[f"{colprefix}volatility_ui"] = UlcerIndex(
            close=df[close], window=14, fillna=fillna
        ).ulcer_index()

    return df


def add_trend_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # MACD
    indicator_macd = MACD(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
    df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # SMAs
    df[f"{colprefix}trend_sma_fast"] = SMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_slow"] = SMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).sma_indicator()

    # EMAs
    df[f"{colprefix}trend_ema_fast"] = EMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).ema_indicator()
    df[f"{colprefix}trend_ema_slow"] = EMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).ema_indicator()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
    )
    df[f"{colprefix}trend_vortex_ind_pos"] = indicator_vortex.vortex_indicator_pos()
    df[f"{colprefix}trend_vortex_ind_neg"] = indicator_vortex.vortex_indicator_neg()
    df[f"{colprefix}trend_vortex_ind_diff"] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    df[f"{colprefix}trend_trix"] = TRIXIndicator(
        close=df[close], window=15, fillna=fillna
    ).trix()

    # Mass Index
    df[f"{colprefix}trend_mass_index"] = MassIndex(
        high=df[high], low=df[low], window_fast=9, window_slow=25, fillna=fillna
    ).mass_index()

    # DPO Indicator
    df[f"{colprefix}trend_dpo"] = DPOIndicator(
        close=df[close], window=20, fillna=fillna
    ).dpo()

    # KST Indicator
    indicator_kst = KSTIndicator(
        close=df[close],
        roc1=10,
        roc2=15,
        roc3=20,
        roc4=30,
        window1=10,
        window2=10,
        window3=10,
        window4=15,
        nsig=9,
        fillna=fillna,
    )
    df[f"{colprefix}trend_kst"] = indicator_kst.kst()
    df[f"{colprefix}trend_kst_sig"] = indicator_kst.kst_sig()
    df[f"{colprefix}trend_kst_diff"] = indicator_kst.kst_diff()

    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(
        high=df[high],
        low=df[low],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=fillna,
    )
    df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
    df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
    df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.ichimoku_a()
    df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.ichimoku_b()

    # Schaff Trend Cycle (STC)
    df[f"{colprefix}trend_stc"] = STCIndicator(
        close=df[close],
        window_slow=50,
        window_fast=23,
        cycle=10,
        smooth1=3,
        smooth2=3,
        fillna=fillna,
    ).stc()

    if not vectorized:
        # Average Directional Movement Index (ADX)
        indicator_adx = ADXIndicator(
            high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
        )
        df[f"{colprefix}trend_adx"] = indicator_adx.adx()
        df[f"{colprefix}trend_adx_pos"] = indicator_adx.adx_pos()
        df[f"{colprefix}trend_adx_neg"] = indicator_adx.adx_neg()

        # CCI Indicator
        df[f"{colprefix}trend_cci"] = CCIIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            window=20,
            constant=0.015,
            fillna=fillna,
        ).cci()

        # Ichimoku Visual Indicator
        indicator_ichi_visual = IchimokuIndicator(
            high=df[high],
            low=df[low],
            window1=9,
            window2=26,
            window3=52,
            visual=True,
            fillna=fillna,
        )
        df[f"{colprefix}trend_visual_ichimoku_a"] = indicator_ichi_visual.ichimoku_a()
        df[f"{colprefix}trend_visual_ichimoku_b"] = indicator_ichi_visual.ichimoku_b()

        # Aroon Indicator
        indicator_aroon = AroonIndicator(close=df[close], window=25, fillna=fillna)
        df[f"{colprefix}trend_aroon_up"] = indicator_aroon.aroon_up()
        df[f"{colprefix}trend_aroon_down"] = indicator_aroon.aroon_down()
        df[f"{colprefix}trend_aroon_ind"] = indicator_aroon.aroon_indicator()

        # PSAR Indicator
        indicator_psar = PSARIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            step=0.02,
            max_step=0.20,
            fillna=fillna,
        )
        # df[f'{colprefix}trend_psar'] = indicator.psar()
        df[f"{colprefix}trend_psar_up"] = indicator_psar.psar_up()
        df[f"{colprefix}trend_psar_down"] = indicator_psar.psar_down()
        df[f"{colprefix}trend_psar_up_indicator"] = indicator_psar.psar_up_indicator()
        df[f"{colprefix}trend_psar_down_indicator"] = indicator_psar.psar_down_indicator()

    return df


def add_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()

    # Stoch RSI (StochRSI)
    indicator_srsi = StochRSIIndicator(
        close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
    df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
    df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    # TSI Indicator
    df[f"{colprefix}momentum_tsi"] = TSIIndicator(
        close=df[close], window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df[f"{colprefix}momentum_uo"] = UltimateOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window1=7,
        window2=14,
        window3=28,
        weight1=4.0,
        weight2=2.0,
        weight3=1.0,
        fillna=fillna,
    ).ultimate_oscillator()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=14,
        smooth_window=3,
        fillna=fillna,
    )
    df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
    df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

    # Williams R Indicator
    df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    ).williams_r()

    # Awesome Oscillator
    df[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # Rate Of Change
    df[f"{colprefix}momentum_roc"] = ROCIndicator(
        close=df[close], window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}momentum_ppo"] = indicator_ppo.ppo()
    df[f"{colprefix}momentum_ppo_signal"] = indicator_ppo.ppo_signal()
    df[f"{colprefix}momentum_ppo_hist"] = indicator_ppo.ppo_hist()

    # Percentage Volume Oscillator
    indicator_pvo = PercentageVolumeOscillator(
        volume=df[volume], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
    df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()
    df[f"{colprefix}momentum_pvo_hist"] = indicator_pvo.pvo_hist()

    if not vectorized:
        # KAMA
        df[f"{colprefix}momentum_kama"] = KAMAIndicator(
            close=df[close], window=10, pow1=2, pow2=30, fillna=fillna
        ).kama()

    return df


def add_others_ta(
    df: pd.DataFrame,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
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
    df[f"{colprefix}others_dr"] = DailyReturnIndicator(
        close=df[close], fillna=fillna
    ).daily_return()

    # Daily Log Return
    df[f"{colprefix}others_dlr"] = DailyLogReturnIndicator(
        close=df[close], fillna=fillna
    ).daily_log_return()

    # Cumulative Return
    df[f"{colprefix}others_cr"] = CumulativeReturnIndicator(
        close=df[close], fillna=fillna
    ).cumulative_return()

    return df


def add_all_ta_features(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
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
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_volatility_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_trend_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_momentum_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
        vectorized=vectorized,
    )
    df = add_others_ta(
        df=df, close=close, fillna=fillna, colprefix=colprefix
    )
    return df


def _job(target: pd.DataFrame, obj: str, kwargs: Dict[str, Union[pd.Series, bool]], jobs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    ohlcv_keys = ["open", "high", "low", "close", "volume"]
    instance = globals()[obj](**{k: target[v] if k in ohlcv_keys else v for k, v in kwargs.items()})
    columns = {}

    for column, attr in jobs:
        values: pd.Series = getattr(instance, attr)()
        columns[column] = values.values

    return columns


def add_all_ta_features_parallel(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted.
        vectorized(bool): if True, use only vectorized functions indicators.
        n_jobs(int): Number of parallel processes to spawn.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    init = [
        # Accumulation Distribution Index
        (
            "AccDistIndexIndicator",
            dict(high=high, low=low, close=close, volume=volume, fillna=fillna),
            [(f"{colprefix}volume_adi", "acc_dist_index")]
        ),
        # On Balance Volume
        (
            "OnBalanceVolumeIndicator",
            dict(close=close, volume=volume, fillna=fillna),
            [(f"{colprefix}volume_obv", "on_balance_volume")]
        ),
        # Chaikin Money Flow
        (
            "ChaikinMoneyFlowIndicator",
            dict(high=high, low=low, close=close, volume=volume, fillna=fillna),
            [(f"{colprefix}volume_cmf", "chaikin_money_flow")]
        ),
        # Force Index
        (
            "ForceIndexIndicator",
            dict(close=close, volume=volume, window=13, fillna=fillna),
            [(f"{colprefix}volume_fi", "force_index")]
        ),
        # Ease of Movement
        (
            "EaseOfMovementIndicator",
            dict(high=high, low=low, volume=volume, window=14, fillna=fillna),
            [
                (f"{colprefix}volume_em", "ease_of_movement"),
                (f"{colprefix}volume_sma_em", "sma_ease_of_movement")
            ]
        ),
        # Volume Price Trend
        (
            "VolumePriceTrendIndicator",
            dict(close=close, volume=volume, fillna=fillna),
            [(f"{colprefix}volume_vpt", "volume_price_trend")]
        ),
        # Volume Weighted Average Price
        (
            "VolumeWeightedAveragePrice",
            dict(high=high, low=low, close=close, volume=volume, window=14, fillna=fillna),
            [(f"{colprefix}volume_vwap", "volume_weighted_average_price")]
        ),

        # Bollinger Bands
        (
            "BollingerBands",
            dict(close=close, window=20, window_dev=2, fillna=fillna),
            [
                (f"{colprefix}volatility_bbm", "bollinger_mavg"),
                (f"{colprefix}volatility_bbh", "bollinger_hband"),
                (f"{colprefix}volatility_bbl", "bollinger_lband"),
                (f"{colprefix}volatility_bbw", "bollinger_wband"),
                (f"{colprefix}volatility_bbp", "bollinger_pband"),
                (f"{colprefix}volatility_bbhi", "bollinger_hband_indicator"),
                (f"{colprefix}volatility_bbli", "bollinger_lband_indicator"),
            ]
        ),
        # Keltner Channel
        (
            "KeltnerChannel",
            dict(close=close, high=high, low=low, window=10, fillna=fillna),
            [
                (f"{colprefix}volatility_kcc", "keltner_channel_mband"),
                (f"{colprefix}volatility_kch", "keltner_channel_hband"),
                (f"{colprefix}volatility_kcl", "keltner_channel_lband"),
                (f"{colprefix}volatility_kcw", "keltner_channel_wband"),
                (f"{colprefix}volatility_kcp", "keltner_channel_pband"),
                (f"{colprefix}volatility_kchi", "keltner_channel_hband_indicator"),
                (f"{colprefix}volatility_kcli", "keltner_channel_lband_indicator"),
            ]
        ),
        # Donchian Channel
        (
            "DonchianChannel",
            dict(high=high, low=low, close=close, window=20, offset=0, fillna=fillna),
            [
                (f"{colprefix}volatility_dcl", "donchian_channel_lband"),
                (f"{colprefix}volatility_dch", "donchian_channel_hband"),
                (f"{colprefix}volatility_dcm", "donchian_channel_mband"),
                (f"{colprefix}volatility_dcw", "donchian_channel_wband"),
                (f"{colprefix}volatility_dcp", "donchian_channel_pband"),
            ]
        ),
        # MACD
        (
            "MACD",
            dict(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna),
            [
                (f"{colprefix}trend_macd", "macd"),
                (f"{colprefix}trend_macd_signal", "macd_signal"),
                (f"{colprefix}trend_macd_diff", "macd_diff"),
            ]
        ),
        # SMAs
        (
            "SMAIndicator",
            dict(close=close, window=12, fillna=fillna),
            [(f"{colprefix}trend_sma_fast", "sma_indicator")],
        ),
        (
            "SMAIndicator",
            dict(close=close, window=26, fillna=fillna),
            [(f"{colprefix}trend_sma_slow", "sma_indicator")],
        ),
        # EMAs
        (
            "EMAIndicator",
            dict(close=close, window=12, fillna=fillna),
            [(f"{colprefix}trend_ema_fast", "ema_indicator")],
        ),
        (
            "EMAIndicator",
            dict(close=close, window=26, fillna=fillna),
            [(f"{colprefix}trend_ema_slow", "ema_indicator")],
        ),
        # Vortex Indicator
        (
            "VortexIndicator",
            dict(high=high, low=low, close=close, window=14, fillna=fillna),
            [
                (f"{colprefix}trend_vortex_ind_pos", "vortex_indicator_pos"),
                (f"{colprefix}trend_vortex_ind_neg", "vortex_indicator_neg"),
                (f"{colprefix}trend_vortex_ind_diff", "vortex_indicator_diff"),
            ]
        ),
        # TRIX Indicator
        (
            "TRIXIndicator",
            dict(close=close, window=15, fillna=fillna),
            [(f"{colprefix}trend_trix", "trix")],
        ),
        # Mass Index
        (
            "MassIndex",
            dict(high=high, low=low, window_fast=9, window_slow=25, fillna=fillna),
            [(f"{colprefix}trend_mass_index", "mass_index")],
        ),
        # DPO Indicator
        (
            "DPOIndicator",
            dict(close=close, window=20, fillna=fillna),
            [(f"{colprefix}trend_dpo", "dpo")],
        ),
        # KST Indicator
        (
            "KSTIndicator",
            dict(
                close=close,
                roc1=10,
                roc2=15,
                roc3=20,
                roc4=30,
                window1=10,
                window2=10,
                window3=10,
                window4=15,
                nsig=9,
                fillna=fillna,
            ),
            [
                (f"{colprefix}trend_kst", "kst"),
                (f"{colprefix}trend_kst_sig", "kst_sig"),
                (f"{colprefix}trend_kst_diff", "kst_diff"),
            ]
        ),
        # Ichimoku Indicator
        (
            "IchimokuIndicator",
            dict(
                high=high,
                low=low,
                window1=9,
                window2=26,
                window3=52,
                visual=False,
                fillna=fillna,
            ),
            [
                (f"{colprefix}trend_ichimoku_conv", "ichimoku_conversion_line"),
                (f"{colprefix}trend_ichimoku_base", "ichimoku_base_line"),
                (f"{colprefix}trend_ichimoku_a", "ichimoku_a"),
                (f"{colprefix}trend_ichimoku_b", "ichimoku_b"),
            ]
        ),
        # Schaff Trend Cycle (STC)
        (
            "STCIndicator",
            dict(
                close=close,
                window_slow=50,
                window_fast=23,
                cycle=10,
                smooth1=3,
                smooth2=3,
                fillna=fillna,
            ),
            [(f"{colprefix}trend_stc", "stc")]
        ),
        # Relative Strength Index (RSI)
        (
            "RSIIndicator",
            dict(close=close, window=14, fillna=fillna),
            [(f"{colprefix}momentum_rsi", "rsi")],
        ),
        # Stoch RSI (StochRSI)
        (
            "StochRSIIndicator",
            dict(close=close, window=14, smooth1=3, smooth2=3, fillna=fillna),
            [
                (f"{colprefix}momentum_stoch_rsi", "stochrsi"),
                (f"{colprefix}momentum_stoch_rsi_k", "stochrsi_k"),
                (f"{colprefix}momentum_stoch_rsi_d", "stochrsi_d"),
            ]
        ),
        # TSI Indicator
        (
            "TSIIndicator",
            dict(close=close, window_slow=25, window_fast=13, fillna=fillna),
            [(f"{colprefix}momentum_tsi", "tsi")],
        ),
        # Ultimate Oscillator
        (
            "UltimateOscillator",
            dict(
                high=high,
                low=low,
                close=close,
                window1=7,
                window2=14,
                window3=28,
                weight1=4.0,
                weight2=2.0,
                weight3=1.0,
                fillna=fillna,
            ),
            [(f"{colprefix}momentum_uo", "ultimate_oscillator")]
        ),
        # Stoch Indicator
        (
            "StochasticOscillator",
            dict(
                high=high,
                low=low,
                close=close,
                window=14,
                smooth_window=3,
                fillna=fillna,
            ),
            [
                (f"{colprefix}momentum_stoch", "stoch"),
                (f"{colprefix}momentum_stoch_signal", "stoch_signal"),
            ]
        ),
        # Williams R Indicator
        (
            "WilliamsRIndicator",
            dict(high=high, low=low, close=close, lbp=14, fillna=fillna),
            [(f"{colprefix}momentum_wr", "williams_r")],
        ),
        # Awesome Oscillator
        (
            "AwesomeOscillatorIndicator",
            dict(high=high, low=low, window1=5, window2=34, fillna=fillna),
            [(f"{colprefix}momentum_ao", "awesome_oscillator")],
        ),
        # Rate Of Change
        (
            "ROCIndicator",
            dict(close=close, window=12, fillna=fillna),
            [(f"{colprefix}momentum_roc", "roc")],
        ),
        # Percentage Price Oscillator
        (
            "PercentagePriceOscillator",
            dict(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna),
            [
                (f"{colprefix}momentum_ppo", "ppo"),
                (f"{colprefix}momentum_ppo_signal", "ppo_signal"),
                (f"{colprefix}momentum_ppo_hist", "ppo_hist"),
            ]
        ),
        # Percentage Volume Oscillator
        (
            "PercentageVolumeOscillator",
            dict(volume=volume, window_slow=26, window_fast=12, window_sign=9, fillna=fillna),
            [
                (f"{colprefix}momentum_pvo", "pvo"),
                (f"{colprefix}momentum_pvo_signal", "pvo_signal"),
                (f"{colprefix}momentum_pvo_hist", "pvo_hist"),
            ]
        ),
        # Daily Return
        (
            "DailyReturnIndicator",
            dict(close=close, fillna=fillna),
            [(f"{colprefix}others_dr", "daily_return")],
        ),
        # Daily Log Return
        (
            "DailyLogReturnIndicator",
            dict(close=close, fillna=fillna),
            [(f"{colprefix}others_dlr", "daily_log_return")],
        ),
        # Cumulative Return
        (
            "CumulativeReturnIndicator",
            dict(close=close, fillna=fillna),
            [(f"{colprefix}others_cr", "cumulative_return")],
        ),
    ]

    if not vectorized:
        init += [
            # Money Flow Indicator
            (
                "MFIIndicator",
                dict(high=high, low=low, close=close, volume=volume, window=14, fillna=fillna),
                [(f"{colprefix}volume_mfi", "money_flow_index")]
            ),
            # Negative Volume Index
            (
                "NegativeVolumeIndexIndicator",
                dict(close=close, volume=volume, fillna=fillna),
                [(f"{colprefix}volume_nvi", "negative_volume_index")]
            ),
            # Average True Range
            (
                "AverageTrueRange",
                dict(close=close, high=high, low=low, window=10, fillna=fillna),
                [(f"{colprefix}volatility_atr", "average_true_range")]
            ),
            # Ulcer Index
            (
                "UlcerIndex",
                dict(close=close, window=14, fillna=fillna),
                [(f"{colprefix}volatility_ui", "ulcer_index")]
            ),
            # Average Directional Movement Index (ADX)
            (
                "ADXIndicator",
                dict(high=high, low=low, close=close, window=14, fillna=fillna),
                [
                    (f"{colprefix}trend_adx", "adx"),
                    (f"{colprefix}trend_adx_pos", "adx_pos"),
                    (f"{colprefix}trend_adx_neg", "adx_neg"),
                ]
            ),
            # CCI Indicator
            (
                "CCIIndicator",
                dict(
                    high=high,
                    low=low,
                    close=close,
                    window=20,
                    constant=0.015,
                    fillna=fillna,
                ),
                [(f"{colprefix}trend_cci", "cci")]
            ),
            # Ichimoku Visual Indicator
            (
                "IchimokuIndicator",
                dict(
                    high=high,
                    low=low,
                    window1=9,
                    window2=26,
                    window3=52,
                    visual=True,
                    fillna=fillna,
                ),
                [
                    (f"{colprefix}trend_visual_ichimoku_a", "ichimoku_a"),
                    (f"{colprefix}trend_visual_ichimoku_b", "ichimoku_b"),
                ]
            ),
            # Aroon Indicator
            (
                "AroonIndicator",
                dict(close=close, window=25, fillna=fillna),
                [
                    (f"{colprefix}trend_aroon_up", "aroon_up"),
                    (f"{colprefix}trend_aroon_down", "aroon_down"),
                    (f"{colprefix}trend_aroon_ind", "aroon_indicator"),
                ]
            ),
            # PSAR Indicator
            (
                "PSARIndicator",
                dict(
                    high=high,
                    low=low,
                    close=close,
                    step=0.02,
                    max_step=0.20,
                    fillna=fillna,
                ),
                [
                    # (f"{colprefix}trend_psar", "psar"),
                    (f"{colprefix}trend_psar_up", "psar_up"),
                    (f"{colprefix}trend_psar_down", "psar_down"),
                    (f"{colprefix}trend_psar_up_indicator", "psar_up_indicator"),
                    (f"{colprefix}trend_psar_down_indicator", "psar_down_indicator"),
                ]
            ),
            # KAMA
            (
                "KAMAIndicator",
                dict(close=close, window=10, pow1=2, pow2=30, fillna=fillna),
                [(f"{colprefix}momentum_kama", "kama")]
            ),
        ]

    columns_order = df.columns.tolist() + [column for *_, jobs in init for column, _ in jobs]

    results = Parallel(n_jobs=n_jobs)(delayed(_job)(df, *args) for args in init)

    for columns in results:
        for column, values in columns.items():
            df[column] = values

    return df[columns_order]
