import os
import yfinance as yf
import pandas as pd
from ta.volume import VolumeWeightedAveragePrice
data = yf.download(
            tickers = 'TSLA',
            period = "1mo",
            group_by = 'ticker',
            auto_adjust = True,
            prepost = False,
            threads = True,
        )
df = data.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['VWAP'] = VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], n=14, fillna=False
    ).volume_weighted_average_price()
print(df.head())

from ta.plots import PlotlyPlot

pp = PlotlyPlot(
            time=df['Date'],
            close=df['Close'],
            open=df['Open'],
            high=df['High'],
            low=df['Low']
        )
pp.candlestickplot(slider=False, showlegend=False)
pp.addTrace(time=df['Date'],
            indicator_data=df['EMA_9'],
            name="EMA_9",
           showlegend=False)
pp.addTrace(time=df['Date'],
            indicator_data=df['VWAP'],
            name="VWAP")
pp.plot()
pp.subplot(time=df['Date'],
            indicator_data=df['EMA_9'],
            name="EMA_9")
