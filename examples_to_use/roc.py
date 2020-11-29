"""This is a example adding volume features.
"""
import pandas as pd
import ta

# Load data
df = pd.read_csv("../test/data/datas.csv", sep=",")

# Clean nan values
df = ta.utils.dropna(df)

window = 12
df[f"roc_{window}"] = ta.momentum.ROCIndicator(close=df["Close"], window=window).roc()
