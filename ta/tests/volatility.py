import unittest

import pandas as pd

from ta.tests.utils import TestIndicator
from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           average_true_range, donchian_channel_hband,
                           donchian_channel_hband_indicator,
                           donchian_channel_lband,
                           donchian_channel_lband_indicator,
                           donchian_channel_mband, donchian_channel_pband,
                           donchian_channel_wband)


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


class TestDonchianChannel(unittest.TestCase):
    """
    https://www.investopedia.com/terms/d/donchianchannels.asp
    https://docs.google.com/spreadsheets/d/17JWWsxSiAb24BLzncUpccc8hg-03QjVWVXmoRCJ2lME/edit#gid=0
    """

    _filename = 'ta/tests/data/cs-dc.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._indicator = DonchianChannel(close=cls._df['Close'], n=20, fillna=False)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_mavg(self):
        target = 'middle_band'
        result = self._indicator.donchian_channel_mband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband(self):
        target = 'upper_band'
        result = self._indicator.donchian_channel_hband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband(self):
        target = 'lower_band'
        result = self._indicator.donchian_channel_lband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband(self):
        target = 'dc_band_width'
        result = self._indicator.donchian_channel_wband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband(self):
        target = 'dc_percentage'
        result = self._indicator.donchian_channel_pband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator(self):
        target = 'dc_high_indicator'
        result = self._indicator.donchian_channel_hband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator(self):
        target = 'dc_low_indicator'
        result = self._indicator.donchian_channel_lband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_mavg2(self):
        target = 'middle_band'
        result = donchian_channel_mband(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband2(self):
        target = 'upper_band'
        result = donchian_channel_hband(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband2(self):
        target = 'lower_band'
        result = donchian_channel_lband(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband2(self):
        target = 'dc_band_width'
        result = donchian_channel_wband(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband2(self):
        target = 'dc_percentage'
        result = donchian_channel_pband(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator2(self):
        target = 'dc_high_indicator'
        result = donchian_channel_hband_indicator(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator2(self):
        target = 'dc_low_indicator'
        result = donchian_channel_lband_indicator(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
