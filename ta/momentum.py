"""
.. module:: momentum
   :synopsis: Momentum Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin, ema


class RSIIndicator(IndicatorMixin):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    """
    def __init__(self, close: pd.Series, n: int = 14, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up = diff.where(diff > 0, 0.0)
        dn = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._n
        emaup = up.ewm(alpha=1/self._n, min_periods=min_periods, adjust=False).mean()
        emadn = dn.ewm(alpha=1/self._n, min_periods=min_periods, adjust=False).mean()
        rs = emaup / emadn
        self._rsi = pd.Series(np.where(emadn == 0, 100, 100-(100/(1+rs))), index=self._close.index)

    def rsi(self) -> pd.Series:
        """Relative Strength Index (RSI)

        Returns:
            pandas.Series: New feature generated.
        """
        rsi = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi, name='rsi')


class TSIIndicator(IndicatorMixin):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, r: int = 25, s: int = 13, fillna: bool = False):
        self._close = close
        self._r = r
        self._s = s
        self._fillna = fillna
        self._run()

    def _run(self):
        m = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._r
        min_periods_s = 0 if self._fillna else self._s
        m1 = m.ewm(span=self._r, min_periods=min_periods_r, adjust=False).mean().ewm(
            span=self._s, min_periods=min_periods_s, adjust=False).mean()
        m2 = abs(m).ewm(span=self._r, min_periods=min_periods_r, adjust=False).mean().ewm(
            span=self._s, min_periods=min_periods_s, adjust=False).mean()
        self._tsi = m1 / m2
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        """True strength index (TSI)

        Returns:
            pandas.Series: New feature generated.
        """
        tsi = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi, name='tsi')


class UltimateOscillator(IndicatorMixin):
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture
    momentum across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period.
        m(int): medium period.
        len(int): long period.
        ws(float): weight of short BP average for UO.
        wm(float): weight of medium BP average for UO.
        wl(float): weight of long BP average for UO.
        fillna(bool): if True, fill nan values with 50.
    """

    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 s: int = 7,
                 m: int = 14,
                 len: int = 28,
                 ws: float = 4.0,
                 wm: float = 2.0,
                 wl: float = 1.0,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._s = s
        self._m = m
        self._len = len
        self._ws = ws
        self._wm = wm
        self._wl = wl
        self._fillna = fillna
        self._run()

    def _run(self):
        cs = self._close.shift(1)
        tr = self._true_range(self._high, self._low, cs)
        bp = self._close - pd.DataFrame({'low': self._low, 'close': cs}).min(axis=1, skipna=False)
        min_periods_s = 0 if self._fillna else self._s
        min_periods_m = 0 if self._fillna else self._m
        min_periods_len = 0 if self._fillna else self._len
        avg_s = bp.rolling(
            self._s, min_periods=min_periods_s).sum() / tr.rolling(self._s, min_periods=min_periods_s).sum()
        avg_m = bp.rolling(
            self._m, min_periods=min_periods_m).sum() / tr.rolling(self._m, min_periods=min_periods_m).sum()
        avg_l = bp.rolling(
            self._len, min_periods=min_periods_len).sum() / tr.rolling(self._len, min_periods=min_periods_len).sum()
        self._uo = (100.0 * ((self._ws * avg_s) + (self._wm * avg_m) + (self._wl * avg_l))
                    / (self._ws + self._wm + self._wl))

    def uo(self) -> pd.Series:
        """Ultimate Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        uo = self._check_fillna(self._uo, value=50)
        return pd.Series(uo, name='uo')


class StochasticOscillator(IndicatorMixin):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    Args:
        close(pandas.Series): dataset 'Close' column.
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n(int): n period.
        d_n(int): sma period over stoch_k.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 n: int = 14,
                 d_n: int = 3,
                 fillna: bool = False):
        self._close = close
        self._high = high
        self._low = low
        self._n = n
        self._d_n = d_n
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n
        smin = self._low.rolling(self._n, min_periods=min_periods).min()
        smax = self._high.rolling(self._n, min_periods=min_periods).max()
        self._stoch_k = 100 * (self._close - smin) / (smax - smin)

    def stoch(self) -> pd.Series:
        """Stochastic Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        stoch_k = self._check_fillna(self._stoch_k, value=50)
        return pd.Series(stoch_k, name='stoch_k')

    def stoch_signal(self) -> pd.Series:
        """Signal Stochastic Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        min_periods = 0 if self._fillna else self._d_n
        stoch_d = self._stoch_k.rolling(self._d_n, min_periods=min_periods).mean()
        stoch_d = self._check_fillna(stoch_d, value=50)
        return pd.Series(stoch_d, name='stoch_k_signal')


class KAMAIndicator(IndicatorMixin):
    """Kaufman's Adaptive Moving Average (KAMA)

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        pow1(int): number of periods for the fastest EMA constant.
        pow2(int): number of periods for the slowest EMA constant.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30, fillna: bool = False):
        self._close = close
        self._n = n
        self._pow1 = pow1
        self._pow2 = pow2
        self._fillna = fillna
        self._run()

    def _run(self):
        close_values = self._close.values
        vol = pd.Series(abs(self._close - np.roll(self._close, 1)))

        min_periods = 0 if self._fillna else self._n
        ER_num = abs(close_values - np.roll(close_values, self._n))
        ER_den = vol.rolling(self._n, min_periods=min_periods).sum()
        ER = ER_num / ER_den

        sc = ((ER*(2.0/(self._pow1+1)-2.0/(self._pow2+1.0))+2/(self._pow2+1.0)) ** 2.0).values

        self._kama = np.zeros(sc.size)
        n = len(self._kama)
        first_value = True

        for i in range(n):
            if np.isnan(sc[i]):
                self._kama[i] = np.nan
            else:
                if first_value:
                    self._kama[i] = close_values[i]
                    first_value = False
                else:
                    self._kama[i] = self._kama[i-1] + sc[i] * (close_values[i] - self._kama[i-1])

    def kama(self) -> pd.Series:
        """Kaufman's Adaptive Moving Average (KAMA)

        Returns:
            pandas.Series: New feature generated.
        """
        kama = pd.Series(self._kama, index=self._close.index)
        kama = self._check_fillna(kama, value=self._close)
        return pd.Series(kama, name='kama')


class ROCIndicator(IndicatorMixin):
    """Rate of Change (ROC)

    The Rate-of-Change (ROC) indicator, which is also referred to as simply
    Momentum, is a pure momentum oscillator that measures the percent change in
    price from one period to the next. The ROC calculation compares the current
    price with the price “n” periods ago. The plot forms an oscillator that
    fluctuates above and below the zero line as the Rate-of-Change moves from
    positive to negative. As a momentum oscillator, ROC signals include
    centerline crossovers, divergences and overbought-oversold readings.
    Divergences fail to foreshadow reversals more often than not, so this
    article will forgo a detailed discussion on them. Even though centerline
    crossovers are prone to whipsaw, especially short-term, these crossovers
    can be used to identify the overall trend. Identifying overbought or
    oversold extremes comes naturally to the Rate-of-Change oscillator.

    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, n: int = 12, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._roc = ((self._close - self._close.shift(self._n)) / self._close.shift(self._n)) * 100

    def roc(self) -> pd.Series:
        """Rate of Change (ROC)

        Returns:
            pandas.Series: New feature generated.
        """
        roc = self._check_fillna(self._roc)
        return pd.Series(roc, name='roc')


class AwesomeOscillatorIndicator(IndicatorMixin):
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO
    calculates the difference of a 34 Period and 5 Period Simple Moving
    Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used
    to affirm trends or to anticipate possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through
    the central points of the bars (H+L)/2, and subtracted from the 5-period
    simple moving average, graphed across the central points of the bars
    (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2

    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period.
        len(int): long period.
        fillna(bool): if True, fill nan values with -50.
    """

    def __init__(self, high: pd.Series, low: pd.Series, s: int = 5, len: int = 34, fillna: bool = False):
        self._high = high
        self._low = low
        self._s = s
        self._len = len
        self._fillna = fillna
        self._run()

    def _run(self):
        mp = 0.5 * (self._high + self._low)
        min_periods_s = 0 if self._fillna else self._s
        min_periods_len = 0 if self._fillna else self._len
        self._ao = mp.rolling(
            self._s, min_periods=min_periods_s).mean() - mp.rolling(self._len, min_periods=min_periods_len).mean()

    def ao(self) -> pd.Series:
        """Awesome Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        ao = self._check_fillna(self._ao, value=0)
        return pd.Series(ao, name='ao')


class WilliamsRIndicator(IndicatorMixin):
    """Williams %R

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period.
        fillna(bool): if True, fill nan values with -50.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, lbp: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._lbp
        hh = self._high.rolling(self._lbp, min_periods=min_periods).max()  # highest high over lookback period lbp
        ll = self._low.rolling(self._lbp, min_periods=min_periods).min()  # lowest low over lookback period lbp
        self._wr = -100 * (hh - self._close) / (hh - ll)

    def wr(self) -> pd.Series:
        """Williams %R

        Returns:
            pandas.Series: New feature generated.
        """
        wr = self._check_fillna(self._wr, value=-50)
        return pd.Series(wr, name='wr')


class StochRSIIndicator(IndicatorMixin):
    """Stochastic RSI

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi
    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period
        d1(int): moving average of Stochastic RSI
        d2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    """
    def __init__(self, close: pd.Series, n: int = 14, d1: int = 3, d2: int = 3, fillna: bool = False):
        self._close = close
        self._n = n
        self._d1 = d1
        self._d2 = d2
        self._fillna = fillna
        self._run()

    def _run(self):
        self._rsi = RSIIndicator(close=self._close, n=self._n, fillna=self._fillna).rsi()
        lowest_low_rsi = self._rsi.rolling(self._n).min()
        self._stochrsi = (self._rsi - lowest_low_rsi) / (self._rsi.rolling(self._n).max() - lowest_low_rsi)

    def stochrsi(self):
        """Stochastic RSI

        Returns:
            pandas.Series: New feature generated.
        """
        stochrsi = self._check_fillna(self._stochrsi)
        return pd.Series(stochrsi, name='stochrsi')

    def stochrsi_k(self):
        """Stochastic RSI %k

        Returns:
            pandas.Series: New feature generated.
        """
        self.stochrsi_k = self._stochrsi.rolling(self._d1).mean()
        stochrsi_K = self._check_fillna(self.stochrsi_k)
        return pd.Series(stochrsi_K, name='stochrsi_k')

    def stochrsi_d(self):
        """Stochastic RSI %d

        Returns:
            pandas.Series: New feature generated.
        """
        stochrsi_D = self.stochrsi_k.rolling(self._d2).mean()
        stochrsi_D = self._check_fillna(stochrsi_D)
        return pd.Series(stochrsi_D, name='stochrsi_d')


class PercentagePriceOscillator(IndicatorMixin):
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        price(pandas.Series): dataset 'Price' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self,
                 close: pd.Series,
                 n_slow: int = 26,
                 n_fast: int = 12,
                 n_sign: int = 9,
                 fillna: bool = False):
        self._close = close
        self._n_slow = n_slow
        self._n_fast = n_fast
        self._n_sign = n_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        _emafast = ema(self._close, self._n_fast, self._fillna)
        _emaslow = ema(self._close, self._n_slow, self._fillna)
        self._ppo = ((_emafast - _emaslow)/_emaslow) * 100
        self._ppo_signal = ema(self._ppo, self._n_sign, self._fillna)
        self._ppo_hist = self._ppo - self._ppo_signal

    def ppo(self):
        """Percentage Price Oscillator Line

        Returns:
            pandas.Series: New feature generated.
        """
        ppo = self._check_fillna(self._ppo, value=0)
        return pd.Series(ppo, name=f'PPO_{self._n_fast}_{self._n_slow}')

    def ppo_signal(self):
        """Percentage Price Oscillator Signal Line

        Returns:
            pandas.Series: New feature generated.
        """

        ppo_signal = self._check_fillna(self._ppo_signal, value=0)
        return pd.Series(ppo_signal, name=f'PPO_sign_{self._n_fast}_{self._n_slow}')

    def ppo_hist(self):
        """Percentage Price Oscillator Histogram

        Returns:
            pandas.Series: New feature generated.
        """

        ppo_hist = self._check_fillna(self._ppo_hist, value=0)
        return pd.Series(ppo_hist, name=f'PPO_hist_{self._n_fast}_{self._n_slow}')


class PercentageVolumeOscillator(IndicatorMixin):
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self,
                 volume: pd.Series,
                 n_slow: int = 26,
                 n_fast: int = 12,
                 n_sign: int = 9,
                 fillna: bool = False):
        self._volume = volume
        self._n_slow = n_slow
        self._n_fast = n_fast
        self._n_sign = n_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        _emafast = ema(self._volume, self._n_fast, self._fillna)
        _emaslow = ema(self._volume, self._n_slow, self._fillna)
        self._pvo = ((_emafast - _emaslow)/_emaslow) * 100
        self._pvo_signal = ema(self._pvo, self._n_sign, self._fillna)
        self._pvo_hist = self._pvo - self._pvo_signal

    def pvo(self):
        """PVO Line

        Returns:
            pandas.Series: New feature generated.
        """
        pvo = self._check_fillna(self._pvo, value=0)
        return pd.Series(pvo, name=f'PVO_{self._n_fast}_{self._n_slow}')

    def pvo_signal(self):
        """Signal Line

        Returns:
            pandas.Series: New feature generated.
        """

        pvo_signal = self._check_fillna(self._pvo_signal, value=0)
        return pd.Series(pvo_signal, name=f'PVO_sign_{self._n_fast}_{self._n_slow}')

    def pvo_hist(self):
        """Histgram

        Returns:
            pandas.Series: New feature generated.
        """

        pvo_hist = self._check_fillna(self._pvo_hist, value=0)
        return pd.Series(pvo_hist, name=f'PVO_hist_{self._n_fast}_{self._n_slow}')


def rsi(close, n=14, fillna=False):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return RSIIndicator(close=close, n=n, fillna=fillna).rsi()


def tsi(close, r=25, s=13, fillna=False):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://en.wikipedia.org/wiki/True_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return TSIIndicator(close=close, r=r, s=s, fillna=fillna).tsi()


def uo(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0, fillna=False):
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture
    momentum across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period.
        m(int): medium period.
        len(int): long period.
        ws(float): weight of short BP average for UO.
        wm(float): weight of medium BP average for UO.
        wl(float): weight of long BP average for UO.
        fillna(bool): if True, fill nan values with 50.

    Returns:
        pandas.Series: New feature generated.

    """
    return UltimateOscillator(
        high=high, low=low, close=close, s=s, m=m, len=len, ws=ws, wm=wm, wl=wl, fillna=fillna).uo()


def stoch(high, low, close, n=14, d_n=3, fillna=False):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    return StochasticOscillator(high=high, low=low, close=close, n=n, d_n=d_n, fillna=fillna).stoch()


def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
    """Stochastic Oscillator Signal

    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return StochasticOscillator(high=high, low=low, close=close, n=n, d_n=d_n, fillna=fillna).stoch_signal()


def wr(high, low, close, lbp=14, fillna=False):
    """Williams %R

    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period.
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """
    return WilliamsRIndicator(high=high, low=low, close=close, lbp=lbp, fillna=fillna).wr()


def ao(high, low, s=5, len=34, fillna=False):
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO
    calculates the difference of a 34 Period and 5 Period Simple Moving
    Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used
    to affirm trends or to anticipate possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through
    the central points of the bars (H+L)/2, and subtracted from the 5-period
    simple moving average, graphed across the central points of the bars
    (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2

    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period.
        len(int): long period.
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """
    return AwesomeOscillatorIndicator(high=high, low=low, s=s, len=len, fillna=fillna).ao()


def kama(close, n=10, pow1=2, pow2=30, fillna=False):
    """Kaufman's Adaptive Moving Average (KAMA)

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n number of periods for the efficiency ratio.
        pow1(int): number of periods for the fastest EMA constant.
        pow2(int): number of periods for the slowest EMA constant.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return KAMAIndicator(close=close, n=n, pow1=pow1, pow2=pow2, fillna=fillna).kama()


def roc(close, n=12, fillna=False):
    """Rate of Change (ROC)

    The Rate-of-Change (ROC) indicator, which is also referred to as simply
    Momentum, is a pure momentum oscillator that measures the percent change in
    price from one period to the next. The ROC calculation compares the current
    price with the price “n” periods ago. The plot forms an oscillator that
    fluctuates above and below the zero line as the Rate-of-Change moves from
    positive to negative. As a momentum oscillator, ROC signals include
    centerline crossovers, divergences and overbought-oversold readings.
    Divergences fail to foreshadow reversals more often than not, so this
    article will forgo a detailed discussion on them. Even though centerline
    crossovers are prone to whipsaw, especially short-term, these crossovers
    can be used to identify the overall trend. Identifying overbought or
    oversold extremes comes naturally to the Rate-of-Change oscillator.

    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n periods.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    return ROCIndicator(close=close, n=n, fillna=fillna).roc()


def stochrsi(close, n=14, d1=3, d2=3, fillna=False):
    """Stochastic RSI

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period
        d1(int): moving average of Stochastic RSI
        d2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(close=close, n=n, d1=d1, d2=d2, fillna=fillna).stochrsi()


def stochrsi_k(close, n=14, d1=3, d2=3, fillna=False):
    """Stochastic RSI %k

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period
        d1(int): moving average of Stochastic RSI
        d2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(close=close, n=n, d1=d1, d2=d2, fillna=fillna).stochrsi_k()


def stochrsi_d(close, n=14, d1=3, d2=3, fillna=False):
    """Stochastic RSI %d

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period
        d1(int): moving average of Stochastic RSI
        d2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(close=close, n=n, d1=d1, d2=d2, fillna=fillna).stochrsi_d()


def ppo(close, n_slow=26, n_fast=12, n_sign=9, fillna=False):
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        price(pandas.Series): dataset 'Price' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).ppo()


def ppo_signal(close, n_slow=26, n_fast=12, n_sign=9, fillna=False):
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        price(pandas.Series): dataset 'Price' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(
        close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).ppo_signal()


def ppo_hist(close, n_slow=26, n_fast=12, n_sign=9, fillna=False):
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        price(pandas.Series): dataset 'Price' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(
        close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).ppo_hist()


def pvo(volume: pd.Series, n_slow: int = 26, n_fast: int = 12, n_sign: int = 9, fillna: bool = False):
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(volume=volume, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna)
    return indicator.pvo()


def pvo_signal(volume: pd.Series, n_slow: int = 26, n_fast: int = 12, n_sign: int = 9, fillna: bool = False):
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(volume=volume, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna)
    return indicator.pvo_signal()


def pvo_hist(volume: pd.Series, n_slow: int = 26, n_fast: int = 12, n_sign: int = 9, fillna: bool = False):
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        n_slow(int): n period long-term.
        n_fast(int): n period short-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(volume=volume, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna)
    return indicator.pvo_hist()
