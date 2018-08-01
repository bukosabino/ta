# -*- coding: utf-8 -*-
"""
.. module:: others
   :synopsis: Others Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd


def daily_return(close, fillna=False):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dr = (close / close.shift(1)) - 1
    dr *= 100
    if fillna:
        dr = dr.fillna(0)
    return pd.Series(dr, name='d_ret')


def daily_log_return(close, fillna=False):
    """Daily Log Return (DLR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dr = np.log(close).diff()
    dr *= 100
    if fillna:
        dr = dr.fillna(0)
    return pd.Series(dr, name='d_logret')


def cumulative_return(close, fillna=False):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cr = (close / close.iloc[0]) - 1
    cr *= 100
    if fillna:
        cr = cr.fillna(method='backfill')
    return pd.Series(cr, name='cum_ret')
