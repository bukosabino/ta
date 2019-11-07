"""It is a technical analysis library to financial time series datasets.
You can use it to do feature engineering from financial datasets. It is
builded on pandas python library.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
from ta.wrapper import (add_all_ta_features, add_momentum_ta, add_others_ta,
                        add_trend_ta, add_volatility_ta, add_volume_ta)
