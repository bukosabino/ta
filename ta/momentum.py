"""
.. module:: momentum
   :synopsis: Momentum Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin


class RSIIndicator(IndicatorMixin):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp
    """

    def __init__(self, close: pd.Series, n: int = 14, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        which_dn = diff < 0

        up, dn = diff, diff*0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]

        emaup = up.ewm(com=self._n-1, min_periods=0).mean()
        emadn = dn.ewm(com=self._n-1, min_periods=0).mean()
        self._rsi = 100 * emaup / (emaup + emadn)

    def rsi(self) -> pd.Series:
        rsi = self.check_fillna(self._rsi, value=50)
        return pd.Series(rsi, name='rsi')


class MFIIndicator(IndicatorMixin):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    """

    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 volume: pd.Series,
                 n: int = 14,
                 fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            volume(pandas.Series): dataset 'Volume' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        # 1 typical price
        tp = (self._high + self._low + self._close) / 3.0

        # 2 up or down column
        up_down = np.where(tp > tp.shift(1), 1, np.where(tp < tp.shift(1), -1, 0))

        # 3 money flow
        mf = tp * self._volume * up_down

        # 4 positive and negative money flow with n periods
        n_positive_mf = mf.rolling(self._n).apply(lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True)
        n_negative_mf = abs(mf.rolling(self._n).apply(lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True))

        # 5 money flow index
        mr = n_positive_mf / n_negative_mf
        self._mr = (100 - (100 / (1 + mr)))

    def money_flow_index(self) -> pd.Series:
        mr = self.check_fillna(self._mr, value=50)
        return pd.Series(mr, name=f'mfi_{self._n}')


class TSIIndicator(IndicatorMixin):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://en.wikipedia.org/wiki/True_strength_index
    """

    def __init__(self, close: pd.Series, r: int = 25, s: int = 13, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            r(int): high period.
            s(int): low period.
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._r = r
        self._s = s
        self._fillna = fillna
        self._run()

    def _run(self):
        m = self._close - self._close.shift(1, fill_value=self._close.mean())
        m1 = m.ewm(self._r).mean().ewm(self._s).mean()
        m2 = abs(m).ewm(self._r).mean().ewm(self._s).mean()
        self._tsi = m1 / m2
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        tsi = self.check_fillna(self._tsi, value=0)
        return pd.Series(tsi, name='tsi')


class UltimateOscillatorIndicator(IndicatorMixin):
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
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            s(int): short period
            m(int): medium period
            len(int): long period
            ws(float): weight of short BP average for UO
            wm(float): weight of medium BP average for UO
            wl(float): weight of long BP average for UO
            fillna(bool): if True, fill nan values with 50.
        """
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
        min_l_or_pc = self._close.shift(1, fill_value=self._close.mean()).combine(self._low, min)
        max_h_or_pc = self._close.shift(1, fill_value=self._close.mean()).combine(self._high, max)
        bp = self._close - min_l_or_pc
        tr = max_h_or_pc - min_l_or_pc
        avg_s = bp.rolling(self._s, min_periods=0).sum() / tr.rolling(self._s, min_periods=0).sum()
        avg_m = bp.rolling(self._m, min_periods=0).sum() / tr.rolling(self._m, min_periods=0).sum()
        avg_l = bp.rolling(self._len, min_periods=0).sum() / tr.rolling(self._len, min_periods=0).sum()
        self._uo = (100.0 * ((self._ws * avg_s) + (self._wm * avg_m) + (self._wl * avg_l))
                    / (self._ws + self._wm + self._wl))

    def uo(self) -> pd.Series:
        uo = self.check_fillna(self._uo, value=50)
        return pd.Series(uo, name='uo')


class StochIndicator(IndicatorMixin):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """

    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 n: int = 14,
                 d_n: int = 3,
                 fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            n(int): n period.
            d_n(int): sma period over stoch_k
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._high = high
        self._low = low
        self._n = n
        self._d_n = d_n
        self._fillna = fillna
        self._run()

    def _run(self):
        smin = self._low.rolling(self._n, min_periods=0).min()
        smax = self._high.rolling(self._n, min_periods=0).max()
        self._stoch_k = 100 * (self._close - smin) / (smax - smin)

    def stoch(self) -> pd.Series:
        stoch_k = self.check_fillna(self._stoch_k, value=50)
        return pd.Series(stoch_k, name='stoch_k')

    def stoch_signal(self) -> pd.Series:
        stoch_d = self._stoch_k.rolling(self._d_n, min_periods=0).mean()
        stoch_d = self.check_fillna(stoch_d, value=50)
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
    """

    def __init__(self, close: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            pow1(int): number of periods for the fastest EMA constant
            pow2(int): number of periods for the slowest EMA constant
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n = n
        self._pow1 = pow1
        self._pow2 = pow2
        self._fillna = fillna
        self._run()

    def _run(self):
        close_values = self._close.values
        vol = pd.Series(abs(self._close - np.roll(self._close, 1)))

        ER_num = abs(close_values - np.roll(close_values, self._n))
        ER_den = vol.rolling(self._n).sum()
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
        kama = pd.Series(self._kama, index=self._close.index)
        kama = self.check_fillna(kama, value=self._close)
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
    """

    def __init__(self, close: pd.Series, n: int = 12, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._roc = ((self._close - self._close.shift(self._n)) / self._close.shift(self._n)) * 100

    def roc(self) -> pd.Series:
        roc = self.check_fillna(self._roc)
        return pd.Series(roc, name='roc')


class AwesomeOscillatorIndicator(IndicatorMixin):
    """
    """

    def __init__(self, high: pd.Series, low: pd.Series, s: int = 5, len: int = 34, fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            s(int): short period
            len(int): long period
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._s = s
        self._len = len
        self._fillna = fillna
        self._run()

    def _run(self):
        mp = 0.5 * (self._high + self._low)
        self._ao = mp.rolling(self._s, min_periods=0).mean() - mp.rolling(self._len, min_periods=0).mean()

    def ao(self) -> pd.Series:
        ao = self.check_fillna(self._ao, value=0)
        return pd.Series(ao, name='ao')


class WilliamsRIndicator(IndicatorMixin):
    """Williams %R

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, lbp: int = 14, fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            lbp(int): lookback period
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._fillna = fillna
        self._run()

    def _run(self):
        hh = self._high.rolling(self._lbp, min_periods=0).max()  # highest high over lookback period lbp
        ll = self._low.rolling(self._lbp, min_periods=0).min()  # lowest low over lookback period lbp
        self._wr = -100 * (hh - self._close) / (hh - ll)

    def wr(self) -> pd.Series:
        wr = self.check_fillna(self._wr, value=-50)
        return pd.Series(wr, name='wr')


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


def money_flow_index(high, low, close, volume, n=14, fillna=False):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    indicator = MFIIndicator(high=high, low=low, close=close, volume=volume, n=n, fillna=fillna)
    return indicator.money_flow_index()


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
        s(int): short period
        m(int): medium period
        len(int): long period
        ws(float): weight of short BP average for UO
        wm(float): weight of medium BP average for UO
        wl(float): weight of long BP average for UO
        fillna(bool): if True, fill nan values with 50.

    Returns:
        pandas.Series: New feature generated.

    """
    return UltimateOscillatorIndicator(
        high=high, low=low, close=close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0, fillna=fillna).uo()


def stoch(high, low, close, n=14, fillna=False):
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
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    return StochIndicator(high=high, low=low, close=close, n=n, d_n=3, fillna=fillna).stoch()


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
    return StochIndicator(high=high, low=low, close=close, n=n, d_n=d_n, fillna=fillna).stoch_signal()


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
        lbp(int): lookback period
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
        s(int): short period
        len(int): long period
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """
    return AwesomeOscillatorIndicator(high=high, low=low, s=s, len=34, fillna=fillna).ao()


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
        close(pandas.Series): dataset 'Close' column
        n(int): n number of periods for the efficiency ratio
        pow1(int): number of periods for the fastest EMA constant
        pow2(int): number of periods for the slowest EMA constant
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
    return ROCIndicator(close=df[close], n=12, fillna=fillna).roc()
