import math

import numpy as np
import pandas as pd


class IndicatorMixin():

    def check_fillna(self, serie: pd.Series, method: str = '', value: int = 0):
        """
        """
        if self._fillna:
            serie_output = serie.copy(deep=False)
            serie_output = serie.replace([np.inf, -np.inf], np.nan)
            serie_output = serie_output.fillna(method='backfill') if method else serie_output.fillna(value)
            return serie_output
        else:
            return serie


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0, adjust=False).mean()
    return series.ewm(span=periods, min_periods=periods, adjust=False).mean()


def get_min_max(x1, x2, f='min'):
    """ Find min or max value between two lists for each index
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
