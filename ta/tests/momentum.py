import unittest

import pandas as pd

from ta.momentum import ROCIndicator, RSIIndicator, roc
from ta.tests.utils import TestIndicator


class TestRateOfChangeIndicator(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    """

    _filename = 'ta/tests/data/cs-roc.csv'

    def test_roc(self):
        target = 'ROC'
        result = roc(close=self._df['Close'], n=12, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_roc2(self):
        target = 'ROC'
        result = ROCIndicator(close=self._df['Close'], n=12, fillna=False).roc()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestRSIIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Note: Using a more simple initilization (directly `ewm`; stockcharts uses `sma` + `ewm`)
    """

    _filename = 'ta/tests/data/cs-rsi.csv'

    def setUp(self):
        self._df = pd.read_csv(self._filename, sep=',')
        self._indicator = RSIIndicator(close=self._df['Close'], n=14, fillna=False)

    def tearDown(self):
        del(self._df)

    def test_rsi(self):
        target = 'RSI'
        result = self._indicator.rsi()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
