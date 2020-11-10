import plotly.graph_objects as go
import pandas as pd

class PlotlyPlot():
    """Plotly Plot

    Plot the stock data as well as indicator data.

    Args:
        time(pandas.Series): dataset 'Timestamp' column.
        close(pandas.Series): dataset 'Close' column.
        open(pandas.Series): dataset 'Open' column.
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        excludeMissing(bool): exclude missing dates.
    """

    def __init__(self,
                 time: pd.Series,
                 close: pd.Series,
                 open: pd.Series = None,
                 high: pd.Series = None,
                 low: pd.Series = None,
                 excludeMissing: bool = False):
        self._time = time
        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._fig = None
        self._excludeMissing = excludeMissing
        self._getlayout()

    def _getlayout(self):
        margin = go.layout.Margin(
            b=50,
            t=50
        )
        if self._excludeMissing:
            self._layout = go.Layout(
                xaxis=dict(type='category'),
                margin=margin
            )
        else:
            self._layout = go.Layout(
                margin=margin
            )

    def candlestickplot(self,
                        increasing_line_color: str = 'green',
                        decreasing_line_color: str = 'red',
                        slider: bool = True,
                        showlegend: bool = True
                        ):
        """Candlestick Plot

        Create the candlestick chart.

        Args:
            increasing_line_color(str): single candlestick color of increasing pattern.
            decreasing_line_color(str): single candlestick color of decreasing pattern.
            slider(bool): if True, show the slider.
            showlegend(bool): if True, show the legend
        """
        self._fig = go.Figure(
            data=[go.Candlestick(
                x=self._time,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                name="Candlestick",
                increasing_line_color=increasing_line_color,
                decreasing_line_color=decreasing_line_color,
                showlegend=showlegend
            )],
            layout=self._layout
        )
        self._fig.update_layout(xaxis_rangeslider_visible=slider)

    def lineplot(self,
                 slider: str = True,
                 showlegend: bool = True
                 ):
        """Line Plot

        Create the close price line chart.

        Args:
            slider(bool): if True, show the slider.
            showlegend(bool): if True, show the legend
        """
        self._fig = go.Figure(
            data=[go.Scatter(
                x=self._time,
                y=self._close,
                name="Close",
                showlegend=showlegend
            )],
            layout=self._layout
        )
        self._fig.update_layout(xaxis_rangeslider_visible=slider)

    def ohlcplot(self,
                 increasing_line_color: str = 'green',
                 decreasing_line_color: str = 'red',
                 slider: str = True,
                 showlegend: bool = True
                 ):
        """OHLC Plot

        Create the OHLC chart

        Args:
            increasing_line_color(str): single candlestick color of increasing pattern.
            decreasing_line_color(str): single candlestick color of decreasing pattern.
            slider(bool): if True, show the slider.
            showlegend(bool): if True, show the legend
        """
        self._fig = go.Figure(
            data=[go.Ohlc(
                x=self._time,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                name="OHLC",
                increasing_line_color=increasing_line_color,
                decreasing_line_color=decreasing_line_color,
                showlegend=showlegend
            )],
            layout=self._layout
        )
        self._fig.update_layout(xaxis_rangeslider_visible=slider)

    def plot(self):
        """Plot

        Ploting existing figure
        """
        self._fig.show()

    def addTrace(self,
                 time: pd.Series,
                 indicator_data: pd.Series,
                 name: str = "",
                 showlegend: bool = True
                 ):
        """Add Trace

        Adding indicator data to the main stock figure

        Args:
            time(pandas.Series): dataset 'Timestamp' column.
            indicator_data(pandas.Series): dataset 'indicator_data' column.
            name(str): name of the indicator
            slider(bool): if True, show the slider.
            showlegend(bool): if True, show the legend
        """
        self._fig.add_trace(
            go.Scatter(
                x=time,
                y=indicator_data,
                name=name,
                showlegend=showlegend
            )
        )

    def subplot(self,
                time: pd.Series,
                indicator_data: pd.Series,
                name: str = "",
                showlegend: bool = True,
                height: int = 200
                ):
        """Subplot

        Create a separate of plot of indicator data

        Args:
            time(pandas.Series): dataset 'Timestamp' column.
            indicator_data(pandas.Series): dataset 'indicator_data' column.
            name(str): name of the indicator
            showlegend(bool): if True, show the legend
            height(int): height of the plot
        """
        line = go.Scatter(
            x=time,
            y=indicator_data,
            name=name,
            showlegend=showlegend
        )
        layout = go.Layout(
            height=height,
            margin=go.layout.Margin(
                b=50,
                t=50,
            )
        )
        fig = go.Figure(data=[line], layout=layout)
        if self._excludeMissing:
            fig.layout.xaxis.type = 'category'
        fig.show()
