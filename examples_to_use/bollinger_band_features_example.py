"""This is a example adding bollinger band features.
"""
import pandas as pd
import ta

# Load data
df = pd.read_csv("../test/data/datas.csv", sep=",")

# Clean nan values
df = ta.utils.dropna(df)

print(df.columns)

# Add bollinger band high indicator filling nans values
df["bb_high_indicator"] = ta.volatility.bollinger_hband_indicator(
    df["Close"], window=20, window_dev=2, fillna=True
)

# Add bollinger band low indicator filling nans values
df["bb_low_indicator"] = ta.volatility.bollinger_lband_indicator(
    df["Close"], window=20, window_dev=2, fillna=True
)

print(df.columns)
