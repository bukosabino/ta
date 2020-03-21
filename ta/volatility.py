"""
.. module:: volatility
   :synopsis: Volatility Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin, ema


class AverageTrueRange(IndicatorMixin):
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
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
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
        atr[self._n-1] = tr[0:self._n].mean()
        for i in range(self._n, len(atr)):
            atr[i] = (atr[i-1] * (self._n-1) + tr.iloc[i]) / float(self._n)
        self._atr = pd.Series(data=atr, index=tr.index)

    def average_true_range(self) -> pd.Series:
        """Average True Range (ATR)

        Returns:
            pandas.Series: New feature generated.
        """
        atr = self._check_fillna(self._atr, value=0)
        return pd.Series(atr, name='atr')


class BollingerBands(IndicatorMixin):
    """Bollinger Bands

    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, n: int = 20, ndev: int = 2, fillna: bool = False):
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
        """Bollinger Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name='mavg')

    def bollinger_hband(self) -> pd.Series:
        """Bollinger Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name='hband')

    def bollinger_lband(self) -> pd.Series:
        """Bollinger Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name='lband')

    def bollinger_wband(self) -> pd.Series:
        """Bollinger Channel Band Width

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name='bbiwband')

    def bollinger_pband(self) -> pd.Series:
        """Bollinger Channel Percentage Band

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name='bbipband')

    def bollinger_hband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing High Band (binary).

        It returns 1, if close is higher than bollinger_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index)
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name='bbihband')

    def bollinger_lband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing Low Band (binary).

        It returns 1, if close is lower than bollinger_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index)
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name='bbilband')


class KeltnerChannel(IndicatorMixin):
    """KeltnerChannel

    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and
    channel direction. Channels can also be used to identify overbought and oversold levels when the trend
    is flat.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
        ov(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    """

    def __init__(
            self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False,
            ov: bool = True):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._ov = ov
        self._run()

    def _run(self):
        if self._ov:
            self._tp = ((self._high + self._low + self._close) / 3.0).rolling(self._n, min_periods=0).mean()
            self._tp_high = (((4 * self._high) - (2 * self._low) + self._close) / 3.0).rolling(
                self._n, min_periods=0).mean()
            self._tp_low = (((-2 * self._high) + (4 * self._low) + self._close) / 3.0).rolling(
                self._n, min_periods=0).mean()
        else:
            self._tp = self._close.ewm(span=self._n, min_periods=0, adjust=False).mean()
            atr = AverageTrueRange(
                close=self._close, high=self._high, low=self._high, n=10, fillna=self._fillna
            ).average_true_range()
            self._tp_high = self._tp + (2*atr)
            self._tp_low = self._tp - (2*atr)

    def keltner_channel_mband(self) -> pd.Series:
        """Keltner Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp = self._check_fillna(self._tp, value=-1)
        return pd.Series(tp, name='mavg')

    def keltner_channel_hband(self) -> pd.Series:
        """Keltner Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp = self._check_fillna(self._tp_high, value=-1)
        return pd.Series(tp, name='kc_hband')

    def keltner_channel_lband(self) -> pd.Series:
        """Keltner Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_low = self._check_fillna(self._tp_low, value=-1)
        return pd.Series(tp_low, name='kc_lband')

    def keltner_channel_wband(self) -> pd.Series:
        """Keltner Channel Band Width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._tp_high - self._tp_low) / self._tp) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name='bbiwband')

    def keltner_channel_pband(self) -> pd.Series:
        """Keltner Channel Percentage Band

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._tp_low) / (self._tp_high - self._tp_low)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name='bbipband')

    def keltner_channel_hband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing High Band (binary)

        It returns 1, if close is higher than keltner_channel_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(np.where(self._close > self._tp_high, 1.0, 0.0), index=self._close.index)
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, name='dcihband')

    def keltner_channel_lband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing Low Band (binary)

        It returns 1, if close is lower than keltner_channel_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(np.where(self._close < self._tp_low, 1.0, 0.0), index=self._close.index)
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name='dcilband')


class DonchianChannel(IndicatorMixin):
    """Donchian Channel

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, n: int = 20, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._hband = self._close.rolling(self._n, min_periods=0).max()
        self._lband = self._close.rolling(self._n, min_periods=0).min()

    def donchian_channel_hband(self) -> pd.Series:
        """Donchian Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name='dchband')

    def donchian_channel_lband(self) -> pd.Series:
        """Donchian Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name='dclband')

    def donchian_channel_hband_indicator(self) -> pd.Series:
        """Donchian Channel Indicator Crossing High Band (binary)

        It returns 1, if close is higher than donchian_channel_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(np.where(self._close >= self._hband, 1.0, 0.0), index=self._close.index)
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, name='dcihband')

    def donchian_channel_lband_indicator(self) -> pd.Series:
        """Donchian Channel Indicator Crossing Low Band (binary)

        It returns 1, if close is lower than donchian_channel_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(np.where(self._close <= self._lband, 1.0, 0.0), index=self._close.index)
        lband = self._check_fillna(lband, value=0)
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


def keltner_channel_mband(high, low, close, n=10, fillna=False):
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
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, fillna=False)
    return indicator.keltner_channel_mband()


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
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, fillna=False)
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
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, fillna=False)
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
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, fillna=False)
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
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, fillna=False)
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
