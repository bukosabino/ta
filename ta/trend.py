"""
.. module:: trend
   :synopsis: Trend Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from .utils import IndicatorMixin, ema, get_min_max


class AroonIndicator(IndicatorMixin):
    """Aroon Indicator

    Identify when trends are likely to change direction.

    Aroon Up - ((N - Days Since N-day High) / N) x 100
    Aroon Down - ((N - Days Since N-day Low) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    """

    def __init__(self, close : pd.Series, n : int = 25, fillna : bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self.close = close
        self.n = n
        self.fillna = fillna

        rolling_close = self.close.rolling(self.n, min_periods=0)
        self.aroon_up_ = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self.n * 100, raw=True)
        self.aroon_down_ = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self.n * 100, raw=True)

    def aroon_up(self) -> pd.Series:
        aroon_up = self.check_fillna(self.aroon_up_, value=0)
        return pd.Series(aroon_up, name=f'aroon_up_{self.n}')

    def aroon_down(self) -> pd.Series:
        aroon_down = self.check_fillna(self.aroon_down_, value=0)
        return pd.Series(aroon_down, name=f'aroon_down_{self.n}')

    def aroon_indicator(self) -> pd.Series:
        aroon_diff = self.aroon_up_ - self.aroon_down_
        aroon_diff = self.check_fillna(aroon_diff, value=0)
        return pd.Series(aroon_diff, name=f'aroon_ind_{self.n}')


class MACD(IndicatorMixin):
    """
    """
    def __init__(self, close : pd.Series, n_slow : int = 12, n_fast : int = 26, n_sign : int = 9, fillna : bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n_fast(int): n period short-term.
            n_slow(int): n period long-term.
            n_sign(int): n period to signal.
            fillna(bool): if True, fill nan values.
        """
        self.close = close
        self.n_slow = n_slow
        self.n_fast = n_fast
        self.n_sign = n_sign
        self.fillna = fillna

        self.emafast = ema(self.close, self.n_fast, self.fillna)
        self.emaslow = ema(self.close, self.n_slow, self.fillna)
        self.macd_ = self.emafast - self.emaslow
        self.macd_signal_ = ema(self.macd_, self.n_sign, self.fillna)
        self.macd_diff_ = self.macd_ - self.macd_signal_

    def macd(self) -> pd.Series:
        macd = self.check_fillna(self.macd_, value=0)
        return pd.Series(macd, name=f'MACD_{self.n_fast}_{self.n_slow}')

    def macd_signal(self) -> pd.Series:
        macd_diff = self.check_fillna(self.macd_signal_, value=0)
        return pd.Series(macd_diff, name=f'MACD_sign_{self.n_fast}_{self.n_slow}')

    def macd_diff(self) -> pd.Series:
        macd_diff = self.check_fillna(self.macd_diff_, value=0)
        return pd.Series(macd_diff, name=f'MACD_diff_{self.n_fast}_{self.n_slow}')


class EMAIndicator(IndicatorMixin):
    """EMA

    Exponential Moving Average
    """

    def __init__(self, close : pd.Series, n : int = 14, fillna : bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self.close = close
        self.n = n
        self.fillna = fillna

    def ema_indicator(self) -> pd.Series:
        """EMA

        Exponential Moving Average

        Returns:
            pandas.Series: New feature generated.
        """
        ema_ = ema(self.close, self.n, self.fillna)
        return pd.Series(ema_, name=f'ema_{self.n}')


class TRIXIndicator(IndicatorMixin):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    """

    def __init__(self, close : pd.Series, n : int = 15, fillna : bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self.close = close
        self.n = n
        self.fillna = fillna

        ema1 = ema(self.close, self.n, self.fillna)
        ema2 = ema(ema1, self.n, self.fillna)
        ema3 = ema(ema2, self.n, self.fillna)
        self.trix_ = (ema3 - ema3.shift(1, fill_value=ema3.mean())) / ema3.shift(1, fill_value=ema3.mean())
        self.trix_ *= 100

    def trix(self) -> pd.Series:
        trix = self.check_fillna(self.trix_, value=0)
        return pd.Series(trix, name=f'trix_{self.n}')


class MassIndex(IndicatorMixin):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    """

    def __init__(self, high : pd.Series, low : pd.Series, n : int = 9, n2 : int = 25, fillna : bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            n(int): n low period.
            n2(int): n high period.
            fillna(bool): if True, fill nan values.
        """
        self.high = high
        self.low = low
        self.n = n
        self.n2 = n2
        self.fillna = fillna

        amplitude = self.high - self.low
        ema1 = ema(amplitude, self.n, self.fillna)
        ema2 = ema(ema1, self.n, self.fillna)
        mass = ema1 / ema2
        self.mass = mass.rolling(self.n2, min_periods=0).sum()

    def mass_index(self) -> pd.Series:
        mass = self.check_fillna(self.mass, value=0)
        return pd.Series(mass, name=f'mass_index_{self.n}_{self.n2}')


class IchimokuIndicator(IndicatorMixin):
    """Ichimoku Kinkō Hyō (Ichimoku)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    """

    def __init__(self, high : pd.Series, low : pd.Series, n1 : int = 9, n2 : int = 26, n3 : int = 52, visual : bool = False, fillna : bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            n1(int): n1 low period.
            n2(int): n2 medium period.
            n3(int): n3 high period.
            visual(bool): if True, shift n2 values.
            fillna(bool): if True, fill nan values.
        """
        self.high = high
        self.low = low
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.visual = visual
        self.fillna = fillna

    def ichimoku_a(self) -> pd.Series:
        conv = 0.5 * (self.high.rolling(self.n1, min_periods=0).max() + self.low.rolling(self.n1, min_periods=0).min())
        base = 0.5 * (self.high.rolling(self.n2, min_periods=0).max() + self.low.rolling(self.n2, min_periods=0).min())
        spana = 0.5 * (conv + base)
        spana = spana.shift(self.n2, fill_value=spana.mean()) if self.visual else spana
        spana = self.check_fillna(spana, method='backfill')
        return pd.Series(spana, name=f'ichimoku_a_{self.n1}_{self.n2}')

    def ichimoku_b(self) -> pd.Series:
        spanb = 0.5 * (self.high.rolling(self.n3, min_periods=0).max() + self.low.rolling(self.n3, min_periods=0).min())
        spanb = spanb.shift(self.n2, fill_value=spanb.mean()) if self.visual else spanb
        spanb = self.check_fillna(spanb, method='backfill')
        return pd.Series(spanb, name=f'ichimoku_b_{self.n1}_{self.n2}')


def ema_indicator(close, n=12, fillna=False):
    """EMA

    Exponential Moving Average

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = EMAIndicator(close=close, n=n, fillna=fillna)
    return indicator.ema_indicator()


def macd(close, n_fast=12, n_slow=26, fillna=False):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=9, fillna=fillna)
    return indicator.macd()


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Signal)

    Shows EMA of MACD.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna)
    return indicator.macd_signal()


def macd_diff(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Diff)

    Shows the relationship between MACD and MACD Signal.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna)
    return indicator.macd_diff()


def ema_indicator(close, n=12, fillna=False):
    """EMA

    Exponential Moving Average via Pandas

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    ema_ = ema(close, n, fillna)
    return pd.Series(ema_, name='ema')


def adx(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'max'))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1]/float(n)) + pos[n+i]

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1]/float(n)) + neg[n+i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i]/trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i]/trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n+1, len(adx)):
        adx[i] = ((adx[i-1] * (n - 1)) + dx[i-1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=close.index)

    if fillna:
        adx = adx.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(adx, name='adx')


def adx_pos(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index Positive (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'max'))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1]/float(n)) + pos[n+i]

    dip = np.zeros(len(close))
    for i in range(1, len(trs)-1):
        dip[i+n] = 100 * (dip_mio[i]/trs[i])

    dip = pd.Series(data=dip, index=close.index)

    if fillna:
        dip = dip.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(dip, name='adx_pos')


def adx_neg(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index Negative (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'max'))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1]/float(n)) + neg[n+i]

    din = np.zeros(len(close))
    for i in range(1, len(trs)-1):
        din[i+n] = 100 * (din_mio[i]/float(trs[i]))

    din = pd.Series(data=din, index=close.index)

    if fillna:
        din = din.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(din, name='adx_neg')


class VortexIndicator(IndicatorMixin):
    """Vortex Indicator (VI)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    """

    def __init__(self, high : pd.Series, low : pd.Series, close : pd.Series, n : int = 14, fillna : bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self.high = high
        self.low = low
        self.close = close
        self.n = n
        self.fillna = fillna

    def vortex_indicator_pos(high, low, close, n=14, fillna=False):
        pass

    def vortex_indicator_neg(high, low, close, n=14, fillna=False):
        pass


def vortex_indicator_pos(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = (high.combine(close.shift(1, fill_value=close.mean()), max)
          - low.combine(close.shift(1, fill_value=close.mean()), min))
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1, fill_value=low.mean()))
    vmm = np.abs(low - high.shift(1, fill_value=high.mean()))

    vip = vmp.rolling(n, min_periods=0).sum() / trn
    if fillna:
        vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vip, name='vip')


def vortex_indicator_neg(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bearish signal triggers when the negative trend indicator
    crosses above the positive trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    if fillna:
        vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vin, name='vin')


def trix(close, n=15, fillna=False):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = TRIXIndicator(close=close, n=n, fillna=fillna)
    return indicator.trix()


def mass_index(high, low, n=9, n2=25, fillna=False):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n(int): n low period.
        n2(int): n high period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    indicator = MassIndex(high=high, low=low, n=n, n2=n2, fillna=fillna)
    return indicator.mass_index()


def cci(high, low, close, n=20, c=0.015, fillna=False):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n periods.
        c(int): constant.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    pp = (high + low + close) / 3.0
    mad = lambda x : np.mean(np.abs(x-np.mean(x)))
    cci = ((pp - pp.rolling(n, min_periods=0).mean())
           / (c * pp.rolling(n, min_periods=0).apply(mad, True)))
    if fillna:
        cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cci, name='cci')


def dpo(close, n=20, fillna=False):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dpo = close.shift(int((0.5 * n) + 1), fill_value=close.mean()) - close.rolling(n, min_periods=0).mean()
    if fillna:
        dpo = dpo.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dpo, name='dpo_'+str(n))


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False):
    """KST Oscillator (KST)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    https://en.wikipedia.org/wiki/KST_oscillator

    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1, fill_value=close.mean()))
              / close.shift(r1, fill_value=close.mean())).rolling(n1, min_periods=0).mean()
    rocma2 = ((close - close.shift(r2, fill_value=close.mean()))
              / close.shift(r2, fill_value=close.mean())).rolling(n2, min_periods=0).mean()
    rocma3 = ((close - close.shift(r3, fill_value=close.mean()))
              / close.shift(r3, fill_value=close.mean())).rolling(n3, min_periods=0).mean()
    rocma4 = ((close - close.shift(r4, fill_value=close.mean()))
              / close.shift(r4, fill_value=close.mean())).rolling(n4, min_periods=0).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    if fillna:
        kst = kst.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst, name='kst')


def kst_sig(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst

    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1, fill_value=close.mean()))
              / close.shift(r1, fill_value=close.mean())).rolling(n1, min_periods=0).mean()
    rocma2 = ((close - close.shift(r2, fill_value=close.mean()))
              / close.shift(r2, fill_value=close.mean())).rolling(n2, min_periods=0).mean()
    rocma3 = ((close - close.shift(r3, fill_value=close.mean()))
              / close.shift(r3, fill_value=close.mean())).rolling(n3, min_periods=0).mean()
    rocma4 = ((close - close.shift(r4, fill_value=close.mean()))
              / close.shift(r4, fill_value=close.mean())).rolling(n4, min_periods=0).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_sig = kst.rolling(nsig, min_periods=0).mean()
    if fillna:
        kst_sig = kst_sig.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst_sig, name='kst_sig')


def ichimoku_a(high, low, n1=9, n2=26, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n1(int): n1 low period.
        n2(int): n2 medium period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = IchimokuIndicator(high=high, low=low, n1=n1, n2=n2, n3=52, visual=visual, fillna=fillna)
    return indicator.ichimoku_a()

    """
    conv = 0.5 * (high.rolling(n1, min_periods=0).max() + low.rolling(n1, min_periods=0).min())
    base = 0.5 * (high.rolling(n2, min_periods=0).max() + low.rolling(n2, min_periods=0).min())

    spana = 0.5 * (conv + base)

    if visual:
        spana = spana.shift(n2, fill_value=spana.mean())

    if fillna:
        spana = spana.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spana, name='ichimoku_a_'+str(n2))
    """


def ichimoku_b(high, low, n2=26, n3=52, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n2(int): n2 medium period.
        n3(int): n3 high period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = IchimokuIndicator(high=high, low=low, n1=9, n2=n2, n3=n3, visual=visual, fillna=fillna)
    return indicator.ichimoku_b()

    """
    spanb = 0.5 * (high.rolling(n3, min_periods=0).max() + low.rolling(n3, min_periods=0).min())

    if visual:
        spanb = spanb.shift(n2, fill_value=spanb.mean())

    if fillna:
        spanb = spanb.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spanb, name='ichimoku_b_'+str(n2))
    """


def aroon_up(close, n=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (uptrend).

    Aroon Up - ((N - Days Since N-day High) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    indicator = AroonIndicator(close=close, n=n, fillna=fillna)
    return indicator.aroon_up()


def aroon_down(close, n=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (downtrend).

    Aroon Down - ((N - Days Since N-day Low) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = AroonIndicator(close=close, n=n, fillna=fillna)
    return indicator.aroon_down()
