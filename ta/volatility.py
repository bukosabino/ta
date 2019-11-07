"""
.. module:: volatility
   :synopsis: Volatility Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin


class AverageTrueRange(IndicatorMixin):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
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
        cs = self._close.shift(1)
        tr = self._high.combine(cs, max) - self._low.combine(cs, min)
        atr = np.zeros(len(self._close))
        atr[0] = tr[1::].mean()
        for i in range(1, len(atr)):
            atr[i] = (atr[i-1] * (self._n-1) + tr.iloc[i]) / float(self._n)
        self._atr = pd.Series(data=atr, index=tr.index)

    def average_true_range(self) -> pd.Series:
        atr = self.check_fillna(self._atr, value=0)
        return pd.Series(atr, name='atr')


class BollingerBands(IndicatorMixin):
    """ Bollinger Bands

        https://en.wikipedia.org/wiki/Bollinger_Bands
    """

    def __init__(self, close: pd.Series, n: int = 20, ndev: int = 2, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            ndev(int): n factor standard deviation
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n = n
        self._ndev = ndev
        self._fillna = fillna
        self._run()

    def _run(self):
        self._mavg = self._close.rolling(self._n, min_periods=0).mean()
        self._mstd = self._close.rolling(self._n, min_periods=0).std(ddof=0)
        self._hband = self._mavg + self._ndev * self._mstd
        self._lband = self._mavg - self._ndev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        mavg = self.check_fillna(self._mavg, method='backfill')
        return pd.Series(mavg, name='mavg')

    def bollinger_hband(self) -> pd.Series:
        hband = self.check_fillna(self._hband, method='backfill')
        return pd.Series(hband, name='hband')

    def bollinger_lband(self) -> pd.Series:
        lband = self.check_fillna(self._lband, method='backfill')
        return pd.Series(lband, name='lband')

    def bollinger_hband_indicator(self) -> pd.Series:
        hband = pd.Series(np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index)
        hband = self.check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name='bbihband')

    def bollinger_lband_indicator(self) -> pd.Series:
        lband = pd.Series(np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index)
        lband = self.check_fillna(lband, value=0)
        return pd.Series(lband, name='bbilband')


class KeltnerChannel(IndicatorMixin):
    """
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
        self._tp = ((self._high + self._low + self._close) / 3.0).rolling(self._n, min_periods=0).mean()
        self._tp_high = (((4 * self._high) - (2 * self._low) + self._close) / 3.0).rolling(
            self._n, min_periods=0).mean()
        self._tp_low = (((-2 * self._high) + (4 * self._low) + self._close) / 3.0).rolling(
            self._n, min_periods=0).mean()

    def keltner_channel_central(self) -> pd.Series:
        tp = self.check_fillna(self._tp, method='backfill')
        return pd.Series(tp, name='mavg')

    def keltner_channel_hband(self) -> pd.Series:
        tp = self.check_fillna(self._tp, method='backfill')
        return pd.Series(tp, name='kc_hband')

    def keltner_channel_lband(self) -> pd.Series:
        tp_low = self.check_fillna(self._tp_low, method='backfill')
        return pd.Series(tp_low, name='kc_lband')

    def keltner_channel_hband_indicator(self) -> pd.Series:
        hband = pd.Series(np.where(self._close > self._tp_high, 1.0, 0.0), index=self._close.index)
        hband = self.check_fillna(hband, value=0)
        return pd.Series(hband, name='dcihband')

    def keltner_channel_lband_indicator(self) -> pd.Series:
        lband = pd.Series(np.where(self._close < self._tp_low, 1.0, 0.0), index=self._close.index)
        lband = self.check_fillna(lband, value=0)
        return pd.Series(lband, name='dcilband')


class DonchianChannel(IndicatorMixin):
    """Donchian Channel

    https://www.investopedia.com/terms/d/donchianchannels.asp

    """

    def __init__(self, close: pd.Series, n: int = 20, fillna: bool = False):
        """
        Args:
            close(pandas.Series): dataset 'Close' column.
            n(int): n period.
            ndev(int): n factor standard deviation
            fillna(bool): if True, fill nan values.
        """
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._hband = self._close.rolling(self._n, min_periods=0).max()
        self._lband = self._close.rolling(self._n, min_periods=0).min()

    def donchian_channel_hband(self) -> pd.Series:
        hband = self.check_fillna(self._hband, method='backfill')
        return pd.Series(hband, name='dchband')

    def donchian_channel_lband(self) -> pd.Series:
        lband = self.check_fillna(self._lband, method='backfill')
        return pd.Series(lband, name='dclband')

    def donchian_channel_hband_indicator(self) -> pd.Series:
        hband = pd.Series(np.where(self._close >= self._hband, 1.0, 0.0), index=self._close.index)
        hband = self.check_fillna(hband, value=0)
        return pd.Series(hband, name='dcihband')

    def donchian_channel_lband_indicator(self) -> pd.Series:
        lband = pd.Series(np.where(self._close <= self._lband, 1.0, 0.0), index=self._close.index)
        lband = self.check_fillna(lband, value=0)
        return pd.Series(lband, name='dcilband')


def average_true_range(high, low, close, n=14, fillna=False):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = AverageTrueRange(high=high, low=low, close=close, n=n, fillna=fillna)
    return indicator.average_true_range()


def bollinger_mavg(close, n=20, fillna=False):
    """Bollinger Bands (BB)

    N-period simple moving average (MA).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, n=n, fillna=fillna)
    return indicator.bollinger_mavg()


def bollinger_hband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Upper band at K times an N-period standard deviation above the moving
    average (MA + Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_hband()


def bollinger_lband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Lower band at K times an N-period standard deviation below the moving
    average (MA − Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_lband()


def bollinger_hband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger High Band Indicator

    Returns 1, if close is higher than bollinger high band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_hband_indicator()


def bollinger_lband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger Low Band Indicator

    Returns 1, if close is lower than bollinger low band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_hband_indicator()


def keltner_channel_central(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (central) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(high=high, low=low, close=close, n=10, fillna=False)
    return indicator.keltner_channel_central()


def keltner_channel_hband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (high) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(high=high, low=low, close=close, n=10, fillna=False)
    return indicator.keltner_channel_hband()


def keltner_channel_lband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (low) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(high=high, low=low, close=close, n=10, fillna=False)
    return indicator.keltner_channel_lband()


def keltner_channel_hband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel High Band Indicator (KC)

    Returns 1, if close is higher than keltner high band channel. Else,
    return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(high=high, low=low, close=close, n=10, fillna=False)
    return indicator.keltner_channel_hband_indicator()


def keltner_channel_lband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel Low Band Indicator (KC)

    Returns 1, if close is lower than keltner low band channel. Else, return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(high=high, low=low, close=close, n=10, fillna=False)
    return indicator.keltner_channel_lband_indicator()


def donchian_channel_hband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The upper band marks the highest price of an issue for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(close=close, n=n, fillna=fillna)
    return indicator.donchian_channel_hband()


def donchian_channel_lband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The lower band marks the lowest price for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(close=close, n=n, fillna=fillna)
    return indicator.donchian_channel_lband()


def donchian_channel_hband_indicator(close, n=20, fillna=False):
    """Donchian High Band Indicator

    Returns 1, if close is higher than donchian high band channel. Else,
    return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(close=close, n=n, fillna=fillna)
    return indicator.donchian_channel_hband_indicator()


def donchian_channel_lband_indicator(close, n=20, fillna=False):
    """Donchian Low Band Indicator

    Returns 1, if close is lower than donchian low band channel. Else,
    return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(close=close, n=n, fillna=fillna)
    return indicator.donchian_channel_lband_indicator()
