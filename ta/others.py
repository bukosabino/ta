"""
.. module:: others
   :synopsis: Others Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from ta.utils import IndicatorMixin


class DailyReturnIndicator(IndicatorMixin):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = (self._close / self._close.shift(1)) - 1
        self._dr *= 100

    def daily_return(self) -> pd.Series:
        """Daily Return (DR)

        Returns:
            pandas.Series: New feature generated.
        """
        dr_series = self._check_fillna(self._dr, value=0)
        return pd.Series(dr_series, name="d_ret")


class DailyLogReturnIndicator(IndicatorMixin):
    """Daily Log Return (DLR)

    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = pd.Series(np.log(self._close)).diff()
        self._dr *= 100

    def daily_log_return(self) -> pd.Series:
        """Daily Log Return (DLR)

        Returns:
            pandas.Series: New feature generated.
        """
        dr_series = self._check_fillna(self._dr, value=0)
        return pd.Series(dr_series, name="d_logret")


class CumulativeReturnIndicator(IndicatorMixin):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._cr = (self._close / self._close.iloc[0]) - 1
        self._cr *= 100

    def cumulative_return(self) -> pd.Series:
        """Cumulative Return (CR)

        Returns:
            pandas.Series: New feature generated.
        """
        cum_ret = self._check_fillna(self._cr, value=-1)
        return pd.Series(cum_ret, name="cum_ret")


class RankIndicator(IndicatorMixin):
    """Rank (IVR-style)

    Computes the rolling rank of the current value within a lookback window,
    expressed as a percentage (0-100). Commonly used for Implied Volatility
    Rank (IVR).

    Rank = (current - min) / (max - min) * 100

    Reference:
        https://www.tastylive.com/concepts-strategies/implied-volatility-rank-percentile

    Args:
        close(pandas.Series): dataset column to rank.
        window(int): lookback period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 252, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        rolling_min = self._close.rolling(window=self._window, min_periods=1).min()
        rolling_max = self._close.rolling(window=self._window, min_periods=1).max()
        denom = rolling_max - rolling_min
        self._rank = np.where(denom != 0, (self._close - rolling_min) / denom * 100, 0)
        self._rank = pd.Series(self._rank, index=self._close.index)

    def rank(self) -> pd.Series:
        """Rank (IVR-style)

        Returns:
            pandas.Series: New feature generated (0-100).
        """
        rank_series = self._check_fillna(self._rank, value=0)
        return pd.Series(rank_series, name="rank")


class PercentileIndicator(IndicatorMixin):
    """Percentile (IVP-style)

    Computes the rolling percentile of the current value within a lookback
    window, expressed as a percentage (0-100). Commonly used for Implied
    Volatility Percentile (IVP).

    Percentile = (number of values below current) / total values * 100

    Reference:
        https://www.tastylive.com/concepts-strategies/implied-volatility-rank-percentile

    Args:
        close(pandas.Series): dataset column to compute percentile for.
        window(int): lookback period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 252, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        def _pct(arr):
            if len(arr) < 2:
                return 0.0
            current = arr.iloc[-1]
            below = (arr.iloc[:-1] < current).sum()
            return below / (len(arr) - 1) * 100

        self._percentile = self._close.rolling(
            window=self._window, min_periods=2
        ).apply(_pct, raw=False)

    def percentile(self) -> pd.Series:
        """Percentile (IVP-style)

        Returns:
            pandas.Series: New feature generated (0-100).
        """
        pct_series = self._check_fillna(self._percentile, value=0)
        return pd.Series(pct_series, name="percentile")


def daily_return(close, fillna=False):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return DailyReturnIndicator(close=close, fillna=fillna).daily_return()


def daily_log_return(close, fillna=False):
    """Daily Log Return (DLR)

    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return DailyLogReturnIndicator(close=close, fillna=fillna).daily_log_return()


def cumulative_return(close, fillna=False):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return CumulativeReturnIndicator(close=close, fillna=fillna).cumulative_return()
