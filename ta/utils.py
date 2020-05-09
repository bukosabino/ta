import math

import numpy as np
import pandas as pd


class IndicatorMixin:

    def _check_fillna(self, serie: pd.Series, value: int = 0):
        """Check if fillna flag is True.

        Args:
            serie(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.

        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            serie_output = serie.copy(deep=False)
            serie_output = serie_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                return serie_output.fillna(method='ffill').fillna(value=-1)
            else:
                return serie_output.fillna(method='ffill').fillna(value)
        else:
            return serie

    def _true_range(self, high: pd.Series, low: pd.Series, prev_close: pd.Series):
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.DataFrame(data={'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df.copy()
    number_cols = df.select_dtypes('number').columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def get_min_max(x1, x2, f='min'):
    """Find min or max value between two lists for each index
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    try:
        if f == 'min':
            return pd.Series(np.amin([x1, x2], axis=0))
        elif f == 'max':
            return pd.Series(np.amax([x1, x2], axis=0))
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    except Exception as e:
        return e
