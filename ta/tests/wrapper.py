import unittest

import pandas as pd

import ta


class TestWrapper(unittest.TestCase):

    _filename = 'ta/tests/data/datas.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_general(self):
        # Clean nan values
        df = ta.utils.dropna(self._df)

        # Add all ta features filling nans values
        ta.add_all_ta_features(
            df=df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=True)

        # Add all ta features not filling nans values
        ta.add_all_ta_features(
            df=df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=False)
