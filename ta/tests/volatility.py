import unittest

import pandas as pd

from ta.tests.utils import TestIndicator
from ta.volatility import AverageTrueRange, BollingerBands, average_true_range


class TestAverageTrueRange(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr
    """

    _filename = 'ta/tests/data/cs-atr.csv'

    def test_atr(self):
        target = 'ATR'
        result = average_true_range(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_atr2(self):
        target = 'ATR'
        result = AverageTrueRange(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14,
            fillna=False).average_true_range()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestBollingerBands(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    """

    _filename = 'ta/tests/data/cs-bbands.csv'

    def setUp(self):
        self._df = pd.read_csv(self._filename, sep=',')
        self._indicator = BollingerBands(close=self._df['Close'], n=20, ndev=2, fillna=False)

    def tearDown(self):
        del(self._df)

    def test_mavg(self):
        target = 'MiddleBand'
        result = self._indicator.bollinger_mavg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband(self):
        target = 'HighBand'
        result = self._indicator.bollinger_hband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband(self):
        target = 'LowBand'
        result = self._indicator.bollinger_lband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband(self):
        target = 'WidthBand'
        result = self._indicator.bollinger_wband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband(self):
        target = 'PercentageBand'
        result = self._indicator.bollinger_pband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator(self):
        target = 'CrossUp'
        result = self._indicator.bollinger_hband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator(self):
        target = 'CrossDown'
        result = self._indicator.bollinger_lband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
