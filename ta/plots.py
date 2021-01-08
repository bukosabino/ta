import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st


class PlotlyPlot:
    """Plotly Plot

    Plot the stock data as well as indicator data.

    Args:
        df(pd.DataFrame): the whole dataset
        time(str): name of dataset 'Date' column.
        close(str): name of dataset 'Close' column.
        open(str): name of dataset 'Open' column.
        high(str): name of dataset 'High' column.
        low(str): name of dataset 'Low' column.
        main_plot_type(str): type of the main plot. Default: "Candlestick"
        main_plot_legend(bool): if True, show the legend of the main plot. Default: False
        range_slider(bool): if True, show the range slider. Default: True
        rows(int): number of plots. Default: 1
        row_height(list): scale for the plots. Default: [1]
    """
    def __init__(self,
                 df: pd.DataFrame,
                 time: str = 'Date',
                 close: str = 'Close',
                 open: str = 'Open',
                 high: str = 'High',
                 low: str = 'Low',
                 main_plot_type: str = "Candlestick",
                 main_plot_legend: bool = False,
                 range_slider: bool = True,
                 rows: int = 1,
                 row_heights: list = None):
        self._time = df[time]
        self._open = df[open]
        self._high = df[high]
        self._low = df[low]
        self._close = df[close]
        self._df = df
        self._chart_type = main_plot_type
        self._showlegend = main_plot_legend
        self._rangeslider = range_slider
        self._rows = rows
        if not row_heights:
            self._row_heights = [1]
        else:
            self._row_heights = row_heights
        self._init_fig()

    def _init_fig(self):
        self._fig = make_subplots(
            rows=self._rows,
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.2,
            row_heights=self._row_heights
        )
        self._main_plot()

    def _main_plot(self):
        data = None
        if self._chart_type == "Candlestick":
            data = go.Candlestick(
                    x=self._time,
                    open=self._open,
                    high=self._high,
                    low=self._low,
                    close=self._close,
                    name="Candlestick",
                    showlegend=self._showlegend
                    )
        elif self._chart_type == "Line":
            data = go.Scatter(
                    x=self._time,
                    y=self._close,
                    name="Close",
                    showlegend=self._showlegend
                    )
        elif self._chart_type == "OHLC":
            data = go.Ohlc(
                    x=self._time,
                    open=self._open,
                    high=self._high,
                    low=self._low,
                    close=self._close,
                    name="OHLC",
                    showlegend=self._showlegend
                )
        self._fig.add_trace(data, row=1, col=1)

    def addLine(self, time, ind_data, name, row, showlegend=True):
        """Add indicator to the plot.

        Add the indicator plot.
        Args:
            time(pandas.Series): dataset 'Timestamp' column.
            ind_data(pandas.Series): dataset indicator column.
            name(str): name of the indicator
            row(int): position of the plot
            showlegend(bool): if True, show the legend
        """
        self._fig.add_trace(go.Scatter(
            x=time,
            y=ind_data,
            name=name,
            showlegend=showlegend), row=row, col=1)

    def addHorizontalLine(self, y, name, row, showlegend=True):
        """Add horizontal line

        Add horizontal line to the plot
        Args:
            y(int): y value of the horizontal line
            name(str): name of the horizontal line
            row(int): position of the horizontal line
            showlegend(bool): if True, show the legend
        """
        self._fig.add_trace(go.Scatter(
            x=self._time,
            y=[y] * len(self._time),
            name=name,
            line=dict(dash='dash', width=0.7),
            showlegend=showlegend), row=row, col=1)

    def show(self):
        """show

        Show the plots
        """
        self._fig.update_layout(
            autosize=True,
            width=700,
            height=700,
            margin={
                'l': 50,
                'r': 50,
                'b': 50,
                't': 50,
                'pad': 4
            },
            paper_bgcolor="LightSteelBlue",
        )
        self._fig.update_layout(
            xaxis_rangeslider_visible=self._rangeslider
        )
        self._fig.show()


class StreamlitPlot(PlotlyPlot):
    """StreamlitPlot

    Plot the stock data as well as indicator data.

    Args:
        df(pd.DataFrame): the whole dataset
        time(str): name of dataset 'Date' column.
        close(str): name of dataset 'Close' column.
        open(str): name of dataset 'Open' column.
        high(str): name of dataset 'High' column.
        low(str): name of dataset 'Low' column.
        main_plot_type(str): type of the main plot. Default: "Candlestick"
        main_plot_legend(bool): if True, show the legend of the main plot. Default: False
        range_slider(bool): if True, show the range slider. Default: True
        rows(int): number of plots. Default: 1
        row_height(list): scale for the plots. Default: [1]
    """
    def __init__(self,
                 df: pd.DataFrame,
                 time: str = 'Date',
                 close: str = 'Close',
                 open: str = 'Open',
                 high: str = 'High',
                 low: str = 'Low',
                 main_plot_type: str = "Candlestick",
                 main_plot_legend: bool = False,
                 range_slider: bool = True,
                 rows: int = 1,
                 row_heights: list = None):
        super().__init__(df, time, close, open, high, low,
                         main_plot_type, main_plot_legend,
                         range_slider, rows, row_heights)

    def _init_fig(self):
        st.title("TA Library Streamlit plot")
        self._fig = make_subplots(
            rows=self._rows,
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.2,
            row_heights=self._row_heights
        )
        self._main_plot()

    def show(self):
        """show

        Show the plots
        """
        self._fig.update_layout(
            autosize=True,
            width=700,
            height=700,
            margin={
                'l': 50,
                'r': 50,
                'b': 50,
                't': 50,
                'pad': 4
            },
            paper_bgcolor="LightSteelBlue",
        )
        self._fig.update_layout(
            xaxis_rangeslider_visible=self._rangeslider
        )
        st.plotly_chart(self._fig)
