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


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()
