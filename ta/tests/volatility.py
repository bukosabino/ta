import pandas as pd

from ta.tests.utils import TestIndicator
from ta.volatility import AverageTrueRange, average_true_range


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
