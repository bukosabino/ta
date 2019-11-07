# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin

import ta


class TAFeaturesTransform(BaseEstimator, TransformerMixin):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    def __init__(self, open_column: str, high_column: str, low_column: str, close_column: str,
                 volume_column: str, fillna: bool = False, colprefix: str = ""):
        self._open_column = open_column
        self._high_column = high_column
        self._low_column = low_column
        self._close_column = close_column
        self._volume_column = volume_column
        self._fillna = fillna
        self._colprefix = colprefix

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = ta.add_all_ta_features(df=X, open=self._open_column, high=self._high_column,
                                   low=self._low_column, close=self._close_column,
                                   volume=self._volume_column, fillna=self._fillna,
                                   colprefix=self._colprefix)

        return X.values
