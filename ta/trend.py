"""
.. module:: trend
   :synopsis: Trend Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin, ema, get_min_max


class AroonIndicator(IndicatorMixin):
    """Aroon Indicator

    Identify when trends are likely to change direction.

    Aroon Up - ((N - Days Since N-day High) / N) x 100
    Aroon Down - ((N - Days Since N-day Low) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    """

    def __init__(self, close: pd.Series, n: int = 25, fillna: bool = False):
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
        rolling_close = self._close.rolling(self._n, min_periods=0)
        self._aroon_up = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self._n * 100, raw=True)
        self._aroon_down = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self._n * 100, raw=True)

    def aroon_up(self) -> pd.Series:
        aroon_up = self.check_fillna(self._aroon_up, value=0)
        return pd.Series(aroon_up, name=f'aroon_up_{self._n}')

    def aroon_down(self) -> pd.Series:
        aroon_down = self.check_fillna(self._aroon_down, value=0)
        return pd.Series(aroon_down, name=f'aroon_down_{self._n}')

    def aroon_indicator(self) -> pd.Series:
        aroon_diff = self._aroon_up - self._aroon_down
        aroon_diff = self.check_fillna(aroon_diff, value=0)
        return pd.Series(aroon_diff, name=f'aroon_ind_{self._n}')


class MACD(IndicatorMixin):
    """
    """
    def __init__(self,
                 close: pd.Series,
                 n_slow: int = 12,
                 n_fast: int = 26,
                 n_sign: int = 9,
                 fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n_fast(int): n period short-term.
            n_slow(int): n period long-term.
            n_sign(int): n period to signal.
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n_slow = n_slow
        self._n_fast = n_fast
        self._n_sign = n_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = ema(self._close, self._n_fast, self._fillna)
        self._emaslow = ema(self._close, self._n_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = ema(self._macd, self._n_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal

    def macd(self) -> pd.Series:
        macd = self.check_fillna(self._macd, value=0)
        return pd.Series(macd, name=f'MACD_{self._n_fast}_{self._n_slow}')

    def macd_signal(self) -> pd.Series:
        macd_diff = self.check_fillna(self._macd_signal, value=0)
        return pd.Series(macd_diff, name=f'MACD_sign_{self._n_fast}_{self._n_slow}')

    def macd_diff(self) -> pd.Series:
        macd_diff = self.check_fillna(self._macd_diff, value=0)
        return pd.Series(macd_diff, name=f'MACD_diff_{self._n_fast}_{self._n_slow}')


class EMAIndicator(IndicatorMixin):
    """EMA

    Exponential Moving Average
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

    def ema_indicator(self) -> pd.Series:
        """EMA

        Exponential Moving Average

        Returns:
            pandas.Series: New feature generated.
        """
        ema_ = ema(self._close, self._n, self._fillna)
        return pd.Series(ema_, name=f'ema_{self._n}')


class TRIXIndicator(IndicatorMixin):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    """

    def __init__(self, close: pd.Series, n: int = 15, fillna: bool = False):
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
        ema1 = ema(self._close, self._n, self._fillna)
        ema2 = ema(ema1, self._n, self._fillna)
        ema3 = ema(ema2, self._n, self._fillna)
        self._trix = (ema3 - ema3.shift(1, fill_value=ema3.mean())) / ema3.shift(1, fill_value=ema3.mean())
        self._trix *= 100

    def trix(self) -> pd.Series:
        trix = self.check_fillna(self._trix, value=0)
        return pd.Series(trix, name=f'trix_{self._n}')


class MassIndex(IndicatorMixin):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    """

    def __init__(self, high: pd.Series, low: pd.Series, n: int = 9, n2: int = 25, fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            n(int): n low period.
            n2(int): n high period.
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._n = n
        self._n2 = n2
        self._fillna = fillna
        self._run()

    def _run(self):
        amplitude = self._high - self._low
        ema1 = ema(amplitude, self._n, self._fillna)
        ema2 = ema(ema1, self._n, self._fillna)
        mass = ema1 / ema2
        self._mass = mass.rolling(self._n2, min_periods=0).sum()

    def mass_index(self) -> pd.Series:
        mass = self.check_fillna(self._mass, value=0)
        return pd.Series(mass, name=f'mass_index_{self._n}_{self._n2}')


class IchimokuIndicator(IndicatorMixin):
    """Ichimoku Kinkō Hyō (Ichimoku)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    """

    def __init__(self, high: pd.Series, low: pd.Series, n1: int = 9, n2: int = 26, n3: int = 52,
                 visual: bool = False, fillna: bool = False):
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
        self._high = high
        self._low = low
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._visual = visual
        self._fillna = fillna

    def ichimoku_a(self) -> pd.Series:
        conv = 0.5 * (self._high.rolling(self._n1, min_periods=0).max()
                      + self._low.rolling(self._n1, min_periods=0).min())
        base = 0.5 * (self._high.rolling(self._n2, min_periods=0).max()
                      + self._low.rolling(self._n2, min_periods=0).min())
        spana = 0.5 * (conv + base)
        spana = spana.shift(self._n2, fill_value=spana.mean()) if self._visual else spana
        spana = self.check_fillna(spana, method='backfill')
        return pd.Series(spana, name=f'ichimoku_a_{self._n1}_{self._n2}')

    def ichimoku_b(self) -> pd.Series:
        spanb = 0.5 * (self._high.rolling(self._n3, min_periods=0).max()
                       + self._low.rolling(self._n3, min_periods=0).min())
        spanb = spanb.shift(self._n2, fill_value=spanb.mean()) if self._visual else spanb
        spanb = self.check_fillna(spanb, method='backfill')
        return pd.Series(spanb, name=f'ichimoku_b_{self._n1}_{self._n2}')


class KSTIndicator(IndicatorMixin):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst
    """

    def __init__(self, close: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30,
                 n1: int = 10, n2: int = 10, n3: int = 10, n4: int = 15, nsig: int = 9,
                 fillna: bool = False):
        """
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
        """
        self._close = close
        self._r1 = r1
        self._r2 = r2
        self._r3 = r3
        self._r4 = r4
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._n4 = n4
        self._nsig = nsig
        self._fillna = fillna
        self._run()

    def _run(self):
        rocma1 = ((self._close - self._close.shift(self._r1, fill_value=self._close.mean()))
                  / self._close.shift(self._r1, fill_value=self._close.mean())).rolling(self._n1, min_periods=0).mean()
        rocma2 = ((self._close - self._close.shift(self._r2, fill_value=self._close.mean()))
                  / self._close.shift(self._r2, fill_value=self._close.mean())).rolling(self._n2, min_periods=0).mean()
        rocma3 = ((self._close - self._close.shift(self._r3, fill_value=self._close.mean()))
                  / self._close.shift(self._r3, fill_value=self._close.mean())).rolling(self._n3, min_periods=0).mean()
        rocma4 = ((self._close - self._close.shift(self._r4, fill_value=self._close.mean()))
                  / self._close.shift(self._r4, fill_value=self._close.mean())).rolling(self._n4, min_periods=0).mean()
        self._kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
        self._kst_sig = self._kst.rolling(self._nsig, min_periods=0).mean()

    def kst(self) -> pd.Series:
        kst = self.check_fillna(self._kst, value=0)
        return pd.Series(kst, name='kst')

    def kst_sig(self) -> pd.Series:
        kst_sig = self.check_fillna(self._kst_sig, value=0)
        return pd.Series(kst_sig, name='kst_sig')

    def kst_diff(self) -> pd.Series:
        kst_diff = self._kst - self._kst_sig
        kst_diff = self.check_fillna(kst_diff, value=0)
        return pd.Series(kst_diff, name='kst_diff')


class DPOIndicator(IndicatorMixin):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    """
    def __init__(self, close: pd.Series, n: int = 20, fillna: bool = False):
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
        self._dpo = (self._close.shift(int((0.5 * self._n) + 1), fill_value=self._close.mean())
                     - self._close.rolling(self._n, min_periods=0).mean())

    def dpo(self) -> pd.Series:
        dpo = self.check_fillna(self._dpo, value=0)
        return pd.Series(dpo, name='dpo_'+str(self._n))


class CCIIndicator(IndicatorMixin):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 n: int = 20,
                 c: float = 0.015,
                 fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            c(int): constant.
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._c = c
        self._fillna = fillna
        self._run()

    def _run(self):

        def _mad(x):
            return np.mean(np.abs(x-np.mean(x)))

        pp = (self._high + self._low + self._close) / 3.0
        self._cci = ((pp - pp.rolling(self._n, min_periods=0).mean())
                     / (self._c * pp.rolling(self._n, min_periods=0).apply(_mad, True)))

    def cci(self) -> pd.Series:
        cci = self.check_fillna(self._cci, value=0)
        return pd.Series(cci, name='cci')


class ADXIndicator(IndicatorMixin):
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
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        assert self._n is not 0, "N may not be 0 and is %r" % n

        cs = self._close.shift(1)
        pdm = get_min_max(self._high, cs, 'max')
        pdn = get_min_max(self._low, cs, 'min')
        tr = pdm - pdn

        self._trs_initial = np.zeros(self._n-1)
        self._trs = np.zeros(len(self._close) - (self._n - 1))
        self._trs[0] = tr.dropna()[0:self._n].sum()
        tr = tr.reset_index(drop=True)

        for i in range(1, len(self._trs)-1):
            self._trs[i] = self._trs[i-1] - (self._trs[i-1]/float(self._n)) + tr[self._n+i]

        up = self._high - self._high.shift(1)
        dn = self._low.shift(1) - self._low
        pos = abs(((up > dn) & (up > 0)) * up)
        neg = abs(((dn > up) & (dn > 0)) * dn)

        self._dip = np.zeros(len(self._close) - (self._n - 1))
        self._dip[0] = pos.dropna()[0:self._n].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(self._dip)-1):
            self._dip[i] = self._dip[i-1] - (self._dip[i-1]/float(self._n)) + pos[self._n+i]

        self._din = np.zeros(len(self._close) - (self._n - 1))
        self._din[0] = neg.dropna()[0:self._n].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(self._din)-1):
            self._din[i] = self._din[i-1] - (self._din[i-1]/float(self._n)) + neg[self._n+i]

    def adx(self) -> pd.Series:
        dip = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            dip[i] = 100 * (self._dip[i]/self._trs[i])

        din = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            din[i] = 100 * (self._din[i]/self._trs[i])

        dx = 100 * np.abs((dip - din) / (dip + din))

        adx = np.zeros(len(self._trs))
        adx[self._n] = dx[0:self._n].mean()

        for i in range(self._n+1, len(adx)):
            adx[i] = ((adx[i-1] * (self._n - 1)) + dx[i-1]) / float(self._n)

        adx = np.concatenate((self._trs_initial, adx), axis=0)
        self._adx = pd.Series(data=adx, index=self._close.index)

        adx = self.check_fillna(self._adx, value=20)
        return pd.Series(adx, name='adx')

    def adx_pos(self) -> pd.Series:
        dip = np.zeros(len(self._close))
        for i in range(1, len(self._trs)-1):
            dip[i+self._n] = 100 * (self._dip[i]/self._trs[i])

        adx_pos = self.check_fillna(pd.Series(dip, index=self._close.index), value=20)
        return pd.Series(adx_pos, name='adx_pos')

    def adx_neg(self) -> pd.Series:
        din = np.zeros(len(self._close))
        for i in range(1, len(self._trs)-1):
            din[i+self._n] = 100 * (self._din[i]/self._trs[i])

        adx_neg = self.check_fillna(pd.Series(din, index=self._close.index), value=20)
        return pd.Series(adx_neg, name='adx_neg')


class VortexIndicator(IndicatorMixin):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
        """
        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            fillna(bool): if True, fill nan values.
        """
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        tr = (self._high.combine(self._close.shift(1, fill_value=self._close.mean()), max)
              - self._low.combine(self._close.shift(1, fill_value=self._close.mean()), min))
        trn = tr.rolling(self._n).sum()
        vmp = np.abs(self._high - self._low.shift(1))
        vmm = np.abs(self._low - self._high.shift(1))
        self._vip = vmp.rolling(self._n, min_periods=0).sum() / trn
        self._vin = vmm.rolling(self._n, min_periods=0).sum() / trn

    def vortex_indicator_pos(self):
        vip = self.check_fillna(self._vip, value=1)
        return pd.Series(vip, name='vip')

    def vortex_indicator_neg(self):
        vin = self.check_fillna(self._vin, value=1)
        return pd.Series(vin, name='vin')

    def vortex_indicator_diff(self):
        vid = self._vip - self._vin
        vid = self.check_fillna(vid, value=0)
        return pd.Series(vid, name='vid')


def ema_indicator(close, n=12, fillna=False):
    """EMA

    Exponential Moving Average

    Returns:
        pandas.Series: New feature generated.
    """
    return EMAIndicator(close=close, n=n, fillna=fillna).ema_indicator()


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
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=9, fillna=fillna).macd()


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
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).macd_signal()


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
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).macd_diff()


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
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx()


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
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx_pos()


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
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx_neg()


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
    return VortexIndicator(high=high, low=low, close=close, n=n, fillna=fillna).vortex_indicator_pos()


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
    return VortexIndicator(high=high, low=low, close=close, n=n, fillna=fillna).vortex_indicator_neg()


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
    return TRIXIndicator(close=close, n=n, fillna=fillna).trix()


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
    return MassIndex(high=high, low=low, n=n, n2=n2, fillna=fillna).mass_index()


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
    return CCIIndicator(high=high, low=low, close=close, n=n, c=c, fillna=fillna).cci()


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
    return DPOIndicator(close=close, n=n, fillna=fillna).dpo()


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
    return KSTIndicator(
        close=close, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, nsig=9, fillna=fillna).kst()


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
    return KSTIndicator(
        close=close, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, nsig=nsig, fillna=fillna).kst_sig()


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
    return IchimokuIndicator(high=high, low=low, n1=n1, n2=n2, n3=52, visual=visual, fillna=fillna).ichimoku_a()


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
    return IchimokuIndicator(high=high, low=low, n1=9, n2=n2, n3=n3, visual=visual, fillna=fillna).ichimoku_b()


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
    return AroonIndicator(close=close, n=n, fillna=fillna).aroon_up()


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
    return AroonIndicator(close=close, n=n, fillna=fillna).aroon_down()
