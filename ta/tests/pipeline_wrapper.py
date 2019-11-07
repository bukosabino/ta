import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ta
from ta.pipeline_wrapper import TAFeaturesTransform
from ta.tests.utils import TestIndicator


class TestTAFeaturesTransform(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx
    """

    _filename = 'ta/tests/data/datas.csv'

    def test_pipeline(self):

        # Settings
        target = 'target'  # name target column
        score = "neg_mean_absolute_error"
        model = ExtraTreesRegressor(n_estimators=50)

        # Nans values
        self._df = ta.utils.dropna(self._df)

        # Labeling
        self._df[target] = (self._df['Close'] / self._df['Close'].shift(1)) - 1

        # Nans values
        self._df = self._df.dropna()

        cols = [i for i in self._df.columns if i != target]
        X = self._df[cols]
        y = self._df[target]

        N = 500

        X_train = self._df[cols][:-N]
        y_train = self._df.target[:-N]
        X_test = self._df[cols][-N:]
        y_test = self._df.target[-N:]

        pipe = make_pipeline(
            TAFeaturesTransform("Open", "High", "Low", "Close", "Volume_BTC", fillna=True),
            StandardScaler(),
            model
        )

        y_pred = pipe.fit(X_train, y_train).predict(X_test)

        # print(mean_absolute_error(y_test, y_pred))
