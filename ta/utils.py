"""
.. module:: utils
   :synopsis: Utils classes and functions.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import math

import numpy as np
import pandas as pd
from numba import guvectorize


class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.

        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.

        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with "Nans" values"""
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


@guvectorize(
    ["void(float64[:], intp[:], float64[:])"],
    "(n),()->(n)",
    nopython=True,
    target="cpu",
)
def numba_sma(arr, window_arr, out):
    """Function to calculate the simple moving average using guvectorize() to create a Numpy ufunc.

    https://numba.readthedocs.io/en/stable/user/vectorize.html?highlight=guvectorize#the-guvectorize-decorator

    https://numba.readthedocs.io/en/stable/user/examples.html?highlight=moving%20average#moving-average

    Modified the numba example so it fills the window size with np.nan

    Args:
        arr (np.array): Numpy array
        window_arr (np.array): Numpy array or simply a int
    Returns:
        guvectorize() functions don’t return their result value: they take it as an array argument, which must be
        filled in by the function. This is because the array is actually allocated by NumPy’s dispatch mechanism,
        which calls into the Numba-generated code. (From the numba docs)
    """
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += arr[i]
        count += 1
        out[i] = np.nan
    out[window_width - 1] = asum / count
    for i in range(window_width, len(arr)):
        asum += arr[i] - arr[i - window_width]
        out[i] = asum / count


@guvectorize(
    ["void(float64[:], intp[:], float64[:])"],
    "(n),()->(n)",
    nopython=True,
    target="cpu",
)
def numba_sma_fillna(arr, window_arr, out):
    """Function to calculate the simple moving average and filling NaN using guvectorize() to create a Numpy ufunc.

    https://numba.readthedocs.io/en/stable/user/vectorize.html?highlight=guvectorize#the-guvectorize-decorator

    https://numba.readthedocs.io/en/stable/user/examples.html?highlight=moving%20average#moving-average

    Slightly modified the numba sma example

    Args:
        arr (np.array): Numpy array
        window_arr (np.array): Numpy array or int
    Returns:
        guvectorize() functions don’t return their result value: they take it as an array argument, which must be
        filled in by the function. This is because the array is actually allocated by NumPy’s dispatch mechanism,
        which calls into the Numba-generated code. (From the numba docs)
    """
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += arr[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(arr)):
        asum += arr[i] - arr[i - window_width]
        out[i] = asum / count


def _sma(series: pd.Series, periods: int, fillna: bool = False) -> pd.Series:
    # the @guvectorize decorator does not work with pylint
    # pylint: disable=locally-disabled, useless-suppression, no-value-for-parameter
    """Helperfunction to use the the fill na functionality, using the two numba guvectorized functions

    Args:
        series (pd.Series): Panda Series.
        periods (int): Window for the simple moving average.
        fillna (bool): If True, fill nan values (default is False).

    Returns:
        pandas.Series: New feature generated.
    """
    series_np_arr = series.to_numpy()
    sma_np_arr = (
        numba_sma_fillna(series_np_arr, periods)
        if fillna
        else numba_sma(series_np_arr, periods)
    )
    return pd.Series(sma_np_arr, index=series.index)


def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)
