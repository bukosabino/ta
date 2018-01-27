# -*- coding: utf-8 -*-
import math
import pandas as pd


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(serie, n, wilder=False):
    """
    https://github.com/FreddieWitherden/ta/blob/master/ta.py
    """
    span = n if not wilder else 2*n - 1
    return serie.ewm(span=span).mean()
