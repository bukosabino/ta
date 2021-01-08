from ta.plots import StreamlitPlot, PlotlyPlot
from ta.momentum import StochasticOscillator
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
indicator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], n=14, d_n=3)
df['Stoch'] = indicator.stoch()
df['Stoch_signal'] = indicator.stoch_signal()

def streamlitPlot_demo(df):
    st_plot = StreamlitPlot(df, 'Date', 'Close', 'Open', 'High', 'Low', rows=2, row_heights=[0.7, 0.3])
    st_plot.addLine(df.index, df['Stoch'], 'Stoch', row=2)
    st_plot.addLine(df.index, df['Stoch_signal'], 'Stoch_signal', row=2)
    st_plot.addHorizontalLine(20, 'oversold', row=2, showlegend=False)
    st_plot.addHorizontalLine(80, 'overbought', row=2, showlegend=False)
    st_plot.show()

def plotlyPlot_demo(df):
    plotly_plot = PlotlyPlot(df, 'Date', 'Close', 'Open', 'High', 'Low', rows=2, row_heights=[0.7, 0.3])
    plotly_plot.addLine(df.index, df['Stoch'], 'Stoch', row=2)
    plotly_plot.addLine(df.index, df['Stoch_signal'], 'Stoch_signal', row=2)
    plotly_plot.addHorizontalLine(20, 'oversold', row=2, showlegend=False)
    plotly_plot.addHorizontalLine(80, 'overbought', row=2, showlegend=False)
    plotly_plot.show()


# streamlitPlot_demo(df)
plotlyPlot_demo(df)