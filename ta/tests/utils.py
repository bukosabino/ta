import unittest
import pandas as pd
import ta


class TestIndicator(unittest.TestCase):

    def setUp(self):
        self._df = pd.read_csv(self._filename, sep=',')

    def tearDown(self):
        del(self._df)


class TestGeneral(TestIndicator):

    _filename = 'ta/tests/data/datas.csv'

    def test_general(self):
        # Clean nan values
        df = ta.utils.dropna(self._df)

        # Add all ta features filling nans values
        ta.add_all_ta_features(self._df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=True)

        # Add all ta features not filling nans values
        df = ta.add_all_ta_features(self._df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=False)
