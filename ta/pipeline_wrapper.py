# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin, BaseEstimator

from .wrapper import *


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

    def __init__(self, open_column, high_column, low_column, close_column,
                     volume_column, fillna=False):
        self.open_column = open_column
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = close_column
        self.volume_column = volume_column
        self.fillna = fillna

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
            
        X = add_all_ta_features(X, self.open_column, self.high_column,
                                self.low_column, self.close_column,
                                self.volume_column, fillna=self.fillna)

        return X.values


