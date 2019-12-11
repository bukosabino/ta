.. Technical Analysis Library in Python documentation master file, created by
   sphinx-quickstart on Tue Apr 10 15:47:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Technical Analysis Library in Python's documentation!
================================================================

It is a Technical Analysis library to financial time series datasets (open, close, high, low, volume). You can use it to do feature engineering from financial datasets. It is builded on Python Pandas library.

Installation (python >= v3.6)
=========================

.. code-block:: bash

    > virtualenv -p python3 virtualenvironment
    > source virtualenvironment/bin/activate
    > pip install ta

Examples
==================

Example adding all features:

.. code-block:: python

    import pandas as pd
    import ta

    # Load datas
    df = pd.read_csv('ta/tests/data/datas.csv', sep=',')

    # Clean NaN values
    df = ta.utils.dropna(df)

    # Add ta features filling NaN values
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=True)


Example adding a particular feature:

.. code-block:: python

   import pandas as pd
   import ta

   # Load datas
   df = pd.read_csv('ta/tests/data/datas.csv', sep=',')

   # Clean NaN values
   df = ta.utils.dropna(df)

   # Initialize Bollinger Bands Indicator
   indicator_bb = ta.volatility.BollingerBands(close=df["Close"], n=20, ndev=2)

   # Add Bollinger Bands features
   df['bb_bbm'] = indicator_bb.bollinger_mavg()
   df['bb_bbh'] = indicator_bb.bollinger_hband()
   df['bb_bbl'] = indicator_bb.bollinger_lband()

   # Add Bollinger Band high indicator
   df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

   # Add Bollinger Band low indicator
   df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

Motivation
==================

* English: https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543
* Spanish: https://medium.com/datos-y-ciencia/biblioteca-de-an%C3%A1lisis-t%C3%A9cnico-sobre-series-temporales-financieras-para-machine-learning-con-cb28f9427d0


Contents
==================
.. toctree::
   TA <ta>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
