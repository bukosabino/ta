import pandas as pd

from ta.tests.utils import TestIndicator
from ta.volume import (EaseOfMovementIndicator, OnBalanceVolumeIndicator,
                       ease_of_movement, on_balance_volume,
                       sma_ease_of_movement)


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


class TestEaseOfMovementIndicator(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv
    """

    _filename = 'ta/tests/data/cs-easeofmovement.csv'

    def test_ease_of_movement(self):
        target = 'EMV'
        result = ease_of_movement(
            high=self._df['High'], low=self._df['Low'], volume=self._df['Volume'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_ease_of_movement2(self):
        target = 'EMV'
        result = EaseOfMovementIndicator(
            high=self._df['High'], low=self._df['Low'], volume=self._df['Volume'], n=14, fillna=False
        ).ease_of_movement()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_sma_ease_of_movement(self):
        target = 'SMA_EMV'
        result = sma_ease_of_movement(
            high=self._df['High'], low=self._df['Low'], volume=self._df['Volume'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_sma_ease_of_movement2(self):
        target = 'SMA_EMV'
        result = EaseOfMovementIndicator(
            high=self._df['High'], low=self._df['Low'], volume=self._df['Volume'], n=14, fillna=False
        ).sma_ease_of_movement()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
