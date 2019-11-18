import pandas as pd

from ta.momentum import ROCIndicator, roc
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
