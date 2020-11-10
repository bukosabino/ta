import os
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.plots import PlotlyPlot

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
df['RSI'] = RSIIndicator(close=df['Close'], n=14).rsi()
indicator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], n=14, d_n=3)
df['Stoch'] = indicator.stoch()
df['Stoch_signal'] = indicator.stoch_signal()

pp = PlotlyPlot(
            time=df['Date'],
            close=df['Close'],
            open=df['Open'],
            high=df['High'],
            low=df['Low']
        )
pp.candlestickplot(showlegend=False)
pp.addTrace(time=df['Date'],
            indicator_data=df['EMA_9'],
            name="EMA_9",
           showlegend=False)
pp.subplot(time=df['Date'],
            indicator_datas=[df['RSI'],df['Stoch'],df['Stoch_signal']],
            names=["RSI",'Stoch','Stoch_signal'],
           positions=[1,2,2],
           row_scale = [0.6,0.2,0.2],
            showlegend=True)
pp.separatePlot(time=df['Date'],
            indicator_data=df['RSI'],
            name="RSI",
            showlegend=True)
