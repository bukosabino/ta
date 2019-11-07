import pandas as pd

from ta.tests.utils import TestIndicator
from ta.volume import OnBalanceVolumeIndicator, on_balance_volume


class TestOnBalanceVolumeIndicator(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    """

    _filename = 'ta/tests/data/cs-obv.csv'

    def test_obv(self):
        target = 'OBV'
        result = on_balance_volume(close=self._df['Close'], volume=self._df['Volume'], fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_obv2(self):
        target = 'OBV'
        result = OnBalanceVolumeIndicator(
            close=self._df['Close'], volume=self._df['Volume'], fillna=False).on_balance_volume()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
