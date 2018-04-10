.. Technical Analysis Library in Python documentation master file, created by
   sphinx-quickstart on Tue Apr 10 15:47:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Technical Analysis Library in Python's documentation!
================================================================

It is a technical analysis library to financial time series datasets (open, close, high, low, volume). You can use it to do feature engineering from financial datasets. It is builded on pandas python library.

Installing
==================

.. code-block:: bash

    > virtualenv -p python3 virtualenvironment
    > source virtualenvironment/bin/activate
    > pip3 install ta

Examples
==================

Adding all features:

.. code-block:: python

   import pandas as pd
   from ta import *

   # Load datas
   df = pd.read_csv('your-file.csv', sep=',')

   # Clean nan values
   df = utils.dropna(df)

   # Add ta features filling Nans values
   df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=True)


Adding individual features:

.. code-block:: python

   import pandas as pd
   from ta import *

   # Load datas
   df = pd.read_csv('your-file.csv', sep=',')

   # Clean nan values
   df = utils.dropna(df)

   # Add bollinger band high indicator filling Nans values
   df['bb_high_indicator'] = bollinger_hband_indicator(df["Close"], n=20, ndev=2, fillna=True)

   # Add bollinger band low indicator filling Nans values
   df['bb_low_indicator'] = bollinger_lband_indicator(df["Close"], n=20, ndev=2, fillna=True)


Contents
==================
.. toctree::
   TA <ta>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
