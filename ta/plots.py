import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,
                         PercentagePriceOscillator, PercentageVolumeOscillator,
                         ROCIndicator, RSIIndicator, StochasticOscillator,
                         StochRSIIndicator, TSIIndicator, UltimateOscillator,
                         WilliamsRIndicator)
from ta.others import (CumulativeReturnIndicator, DailyLogReturnIndicator,
                       DailyReturnIndicator)
from ta.trend import (MACD, ADXIndicator, AroonIndicator, CCIIndicator,
                      DPOIndicator, EMAIndicator, IchimokuIndicator,
                      KSTIndicator, MassIndex, PSARIndicator, SMAIndicator,
                      STCIndicator, TRIXIndicator, VortexIndicator)
from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           KeltnerChannel, UlcerIndex)
from ta.volume import (AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
                       EaseOfMovementIndicator, ForceIndexIndicator,
                       MFIIndicator, NegativeVolumeIndexIndicator,
                       OnBalanceVolumeIndicator, VolumePriceTrendIndicator,
                       VolumeWeightedAveragePrice)


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
        self._data = None
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
                        showlegend: bool = True
                        ):
        """Candlestick Plot

        Create the candlestick chart.

        Args:
            increasing_line_color(str): single candlestick color of increasing pattern.
            decreasing_line_color(str): single candlestick color of decreasing pattern.
            showlegend(bool): if True, show the legend
        """
        self._main_data = [
            go.Candlestick(
                x=self._time,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                name="Candlestick",
                increasing_line_color=increasing_line_color,
                decreasing_line_color=decreasing_line_color,
                showlegend=showlegend)
        ]

    def lineplot(self,
                 showlegend: bool = True
                 ):
        """Line Plot

        Create the close price line chart.

        Args:
            showlegend(bool): if True, show the legend
        """
        self._main_data = [
            go.Scatter(
                x=self._time,
                y=self._close,
                name="Close",
                showlegend=showlegend
            )]

    def ohlcplot(self,
                 increasing_line_color: str = 'green',
                 decreasing_line_color: str = 'red',
                 showlegend: bool = True
                 ):
        """OHLC Plot

        Create the OHLC chart

        Args:
            increasing_line_color(str): single candlestick color of increasing pattern.
            decreasing_line_color(str): single candlestick color of decreasing pattern.
            showlegend(bool): if True, show the legend
        """
        self._main_data = [
            go.Ohlc(
                x=self._time,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                name="OHLC",
                increasing_line_color=increasing_line_color,
                decreasing_line_color=decreasing_line_color,
                showlegend=showlegend
            )
        ]

    def _plot(self, slider, layout):
        if not layout:
            self._fig = go.Figure(
                data=self._main_data,
                layout=self._layout
            )
        else:
            self._fig = go.Figure(
                data=self._main_data,
                layout=layout
            )
        self._fig.update_layout(xaxis_rangeslider_visible=slider)

    def plot(self,
             slider: str = True,
             layout: go.Layout = None):
        """Plot

        Ploting existing figure

        Args:
            slider(bool): if True, show the slider.
            layout(go.Layout): customize layout for the plot.
        """
        self._plot(slider, layout)
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
            showlegend(bool): if True, show the legend
        """
        self._main_data.append(
            go.Scatter(
                x=time,
                y=indicator_data,
                name=name,
                showlegend=showlegend
            )
        )

    def _subplot(self,
                 time: pd.Series,
                 indicator_datas: list,
                 names: list,
                 positions: list,
                 row_scale: list,
                 showlegend: bool = True,
                 layout: go.layout = None,
                 ):
        self._fig = make_subplots(
            rows=len(row_scale),
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.05,
            row_heights=row_scale
        )
        # main plot
        for i in range(len(self._main_data)):
            self._fig.add_trace(self._main_data[i], row=1, col=1)

        # subplot
        for i in range(len(indicator_datas)):
            self._fig.add_trace(go.Scatter(
                x=time,
                y=indicator_datas[i],
                name=names[i],
                showlegend=showlegend
            ), row=positions[i] + 1, col=1
            )
        if not layout:
            self._fig.update_layout(
                height=(500 + (len(row_scale) - 1) * 100),
                xaxis_rangeslider_visible=False
            )
        else:
            self._fig.layout = layout

        if self._excludeMissing:
            self._fig.layout.xaxis.type = 'category'

    def subplot(self,
                time: pd.Series,
                indicator_datas: list,
                names: list,
                positions: list,
                row_scale: list,
                showlegend: bool = True,
                layout: go.layout = None
                ):
        """Subplot

        Create a subplots of plot of indicator data

        Args:
            time(pandas.Series): dataset 'Timestamp' column.
            indicator_datas(list): list of dataset 'indicator_data' columns.
            names(list): list of names of the indicators
            positions(list): list of positions of the subplot of indicator_data
            row_scale(list): list of row scale of the plots
            showlegend(bool): if True, show the legend
            layout(go.Layout): customize layout for the subplot.
        """
        self._subplot(time, indicator_datas, names, positions, row_scale, showlegend, layout)

        self._fig.show()

    def _separatePlot(self,
                      time: pd.Series,
                      indicator_data: pd.Series,
                      name: str = "",
                      showlegend: bool = True,
                      height: int = 200,
                      layout: go.layout = None
                      ):
        line = go.Scatter(
            x=time,
            y=indicator_data,
            name=name,
            showlegend=showlegend
        )
        if not layout:
            layout = go.Layout(
                height=height,
                margin=go.layout.Margin(
                    b=50,
                    t=50,
                )
            )
        self._fig = go.Figure(data=[line], layout=layout)
        if self._excludeMissing:
            self._fig.layout.xaxis.type = 'category'

    def separatePlot(self,
                     time: pd.Series,
                     indicator_data: pd.Series,
                     name: str = "",
                     showlegend: bool = True,
                     height: int = 200,
                     layout: go.layout = None
                     ):
        """Separate Plot

        Create a separate of plot of indicator data

        Args:
            time(pandas.Series): dataset 'Timestamp' column.
            indicator_data(pandas.Series): dataset 'indicator_data' column.
            name(str): name of the indicator
            showlegend(bool): if True, show the legend
            height(int): height of the plot
            layout(go.Layout): customize layout for the separate plot.
        """
        self._separatePlot(time, indicator_data, name, showlegend, height, layout)
        self._fig.show()


ta = {
        "SMA": {
            "params": {
                "Period": 10
            },
            "subplot": False
        },
        "EMA": {
            "params": {
                "Period": 10
            },
            "subplot": False
        },
        "RSI": {
            "params": {
                "Period": 14
            },
            "subplot": True
        },
        "TSI": {
            "params": {
                "High Peroid": 25,
                "Slow Period": 13,
            },
            "subplot": True
        },
        "Stochastic Oscillator": {
            "params": {
                "Period": 14,
                "SMA Period": 3
            },
            "subplot": True
        },
        "Bollinger Bands": {
            "params": {
                "Period": 20,
                "N Factor Standard Deviation": 2
            },
            "subplot": False
        }
    }


class StreamlitPlot:
    """StreamlitPlot

    Plot the stock data as well as indicator data.

    Args:
        df(pd.DataFrame): the whole dataset
        time(str): name of dataset 'Date' column.
        close(str): name of dataset 'Close' column.
        open(str): name of dataset 'Open' column.
        high(str): name of dataset 'High' column.
        low(str): name of dataset 'Low' column.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 time: str = 'Date',
                 close: str = 'Close',
                 open: str = 'Open',
                 high: str = 'High',
                 low: str = 'Low'):
        self._time = df[time]
        self._open = df[open]
        self._high = df[high]
        self._low = df[low]
        self._close = df[close]
        self._df = df
        self._run()

    def _run(self):
        indicator = st.sidebar.selectbox("Technical Indicator", ['']+list(ta.keys()))
        chart_type = st.sidebar.selectbox(
            "Chart Type", ["Candlestick", "Line", "OHLC"])
        st.title("TA Library Streamlit plot")
        if indicator != '':
            params = ta[indicator]['params']

            request_data = {}
            for k, v in params.items():
                if k != "":
                    value = st.sidebar.slider(
                        label=k, min_value=3, max_value=365, value=v)
                    request_data[k] = value
            self._fillna = st.sidebar.checkbox("Fill NA"),
            self._showlegend = st.sidebar.checkbox("Show Legend", value=True)
            self._rangeslider = st.sidebar.checkbox("Range Slider", value=True)
            ind_data = self._graph_plot(
                indicator, ta[indicator]['subplot'], request_data, chart_type)
            for data in ind_data:
                self._df[data.name] = data
        st.dataframe(self._df)

    def _main_plot(self, chart_type):
        data = None
        if chart_type == "Candlestick":
            data = [go.Candlestick(
                    x=self._time,
                    open=self._open,
                    high=self._high,
                    low=self._low,
                    close=self._close,
                    name="Candlestick",
                    showlegend=False
                    )]
        elif chart_type == "Line":
            data = [go.Scatter(
                    x=self._time,
                    y=self._close,
                    name="Close",
                    showlegend=False
                    )]
        elif chart_type == "OHLC":
            data = [
                go.Ohlc(
                    x=self._time,
                    open=self._open,
                    high=self._high,
                    low=self._low,
                    close=self._close,
                    name="OHLC",
                    showlegend=False
                )]
        return data

    def _plot(self, data, ind_data, subplot):
        fig = None
        if subplot:
            fig = make_subplots(
                rows=2,
                shared_xaxes=True,
                shared_yaxes=True,
                cols=1,
                print_grid=False,
                vertical_spacing=0.2,
                row_heights=[0.7, 0.3]
            )
            fig.add_trace(
                data[0], row=1, col=1
            )
            for i in ind_data:
                fig.add_trace(go.Scatter(
                        x=self._time,
                        y=i,
                        name=i.name,
                        showlegend=self._showlegend
                    ), row=2, col=1
                )
        else:
            for i in ind_data:
                data.append(
                    go.Scatter(
                        x=self._time,
                        y=i,
                        name=i.name,
                        showlegend=self._showlegend
                    )
                )
            fig = go.Figure(
                data=data,
                layout=go.Layout(
                    margin=go.layout.Margin(
                        b=50,
                        t=50
                    )
                )
            )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            paper_bgcolor="LightSteelBlue",
        )
        fig.update_layout(xaxis_rangeslider_visible=self._rangeslider)
        st.plotly_chart(fig)

    def _graph_plot(self, indicator, subplot, request_data, chart_type):
        ind_data = self._library_fuct(indicator, request_data)
        data = self._main_plot(chart_type)
        self._plot(data, ind_data, subplot)
        return ind_data
    
    def _library_fuct(self, indicator, params):
        fillna = self._fillna[0]
        if indicator == "SMA":
            return [SMAIndicator(close=self._close, n=params["Period"], fillna=fillna).sma_indicator()]
        elif indicator == "EMA":
            return [EMAIndicator(close=self._close, n=params["Period"], fillna=fillna).ema_indicator()]
        elif indicator == "RSI":
            return [RSIIndicator(close=self._close, n=params["Period"], fillna=fillna).rsi()]
        elif indicator == "TSI":
            return [TSIIndicator(close=self._close,
                                 r=params["High Peroid"],
                                 s=params["Slow Period"],
                                 fillna=fillna).tsi()]
        elif indicator == "Stochastic Oscillator":
            indicator = StochasticOscillator(high=self._high,
                                             low=self._low,
                                             close=self._close,
                                             n=params["Period"],
                                             d_n=params["SMA Period"],
                                             fillna=fillna)
            return [indicator.stoch(), indicator.stoch_signal()]
        elif indicator == "Bollinger Bands":
            indicator_bb = BollingerBands(close=self._close,
                                          n=params["Period"],
                                          ndev=params["N Factor Standard Deviation"],
                                          fillna=fillna)
            return [indicator_bb.bollinger_hband(), indicator_bb.bollinger_lband()]
