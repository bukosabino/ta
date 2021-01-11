from ta.plots import StreamlitPlot, PlotlyPlot
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
import yfinance as yf

def getData():
    data = yf.download(
        tickers='TSLA',
        period="1y",
        group_by='ticker',
        auto_adjust=True,
        prepost=False,
        threads=True,
    )
    return data

df = getData()
df = df.reset_index()

indicator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch'] = indicator.stoch()
df['Stoch_signal'] = indicator.stoch_signal()
indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
df["bbh"] = indicator_bb.bollinger_hband()
df["bbl"] = indicator_bb.bollinger_lband()


def streamlitPlot_demo(df):
    st_plot = StreamlitPlot(df, 'Date', 'Close', 'Open', 'High', 'Low', rows=2, row_heights=[0.7, 0.3])
    st_plot.addLine(df['bbh'], 'bb_high', row=1)
    st_plot.addLine(df['bbl'], 'bb_low', row=1)
    st_plot.addLine(df['Stoch'], 'Stoch', row=2)
    st_plot.addLine(df['Stoch_signal'], 'Stoch_signal', row=2)
    st_plot.addHorizontalLine(20, 'oversold', row=2, showlegend=False, color='blue')
    st_plot.addHorizontalLine(80, 'overbought', row=2, showlegend=False, color='blue')
    st_plot.addHorizontalArea(range=(0, 20), row=2, color='green')
    st_plot.addHorizontalArea(range=(80, 100), row=2, color='red')
    st_plot.show()

def plotlyPlot_demo(df):
    plotly_plot = PlotlyPlot(df, 'Date', 'Close', 'Open', 'High', 'Low', rows=2, row_heights=[0.7, 0.3])
    plotly_plot.addLine(df['Stoch'], 'Stoch', row=2)
    plotly_plot.addLine(df['Stoch_signal'], 'Stoch_signal', row=2)
    plotly_plot.addHorizontalLine(20, 'oversold', row=2, showlegend=False, color='blue')
    plotly_plot.addHorizontalLine(80, 'overbought', row=2, showlegend=False, color='blue')
    plotly_plot.addHorizontalArea(range=(0, 20), row=2, color='green')
    plotly_plot.addHorizontalArea(range=(80, 100), row=2, color='red')
    plotly_plot.show()


streamlitPlot_demo(df)
# plotlyPlot_demo(df)