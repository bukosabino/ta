import unittest

import pandas as pd

from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           KeltnerChannel, average_true_range, bollinger_hband,
                           bollinger_hband_indicator, bollinger_lband,
                           bollinger_lband_indicator, bollinger_mavg,
                           bollinger_pband, bollinger_wband,
                           donchian_channel_hband, donchian_channel_lband,
                           donchian_channel_mband, donchian_channel_pband,
                           donchian_channel_wband, keltner_channel_hband,
                           keltner_channel_hband_indicator,
                           keltner_channel_lband,
                           keltner_channel_lband_indicator,
                           keltner_channel_mband, keltner_channel_pband,
                           keltner_channel_wband)


class TestAverageTrueRange(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr
    https://docs.google.com/spreadsheets/d/1DYG5NI_1px30aZ6oJkDIkWsyJW5V8jGbBVKIr9NWtec/edit?usp=sharing
    """

    _filename = 'ta/tests/data/cs-atr.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=14, fillna=False)
        cls._indicator = AverageTrueRange(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_atr(self):
        target = 'ATR'
        result = average_true_range(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_atr2(self):
        target = 'ATR'
        result = self._indicator.average_true_range()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestAverageTrueRange2(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr
    https://docs.google.com/spreadsheets/d/1IRlmwVmRLAzjIIt2iXBukZyyaSAYB_0iRyAoOowZaBk/edit?usp=sharing
    """

    _filename = 'ta/tests/data/cs-atr2.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=10, fillna=False)
        cls._indicator = AverageTrueRange(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_atr(self):
        target = 'ATR'
        result = self._indicator.average_true_range()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_atr2(self):
        target = 'ATR'
        result = average_true_range(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestBollingerBands(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    """

    _filename = 'ta/tests/data/cs-bbands.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n=20, ndev=2, fillna=False)
        cls._indicator = BollingerBands(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

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

    def test_mavg2(self):
        target = 'MiddleBand'
        result = bollinger_mavg(close=self._df['Close'], n=20, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband2(self):
        target = 'HighBand'
        result = bollinger_hband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband2(self):
        target = 'LowBand'
        result = bollinger_lband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband2(self):
        target = 'WidthBand'
        result = bollinger_wband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband2(self):
        target = 'PercentageBand'
        result = bollinger_pband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator2(self):
        target = 'CrossUp'
        result = bollinger_hband_indicator(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator2(self):
        target = 'CrossDown'
        result = bollinger_lband_indicator(**self._params)
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
        cls._params = dict(
            high=cls._df['high'], low=cls._df['low'], close=cls._df['close'], n=20, offset=0, fillna=False)
        cls._indicator = DonchianChannel(**cls._params)

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

    def test_mavg2(self):
        target = 'middle_band'
        result = donchian_channel_mband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband2(self):
        target = 'upper_band'
        result = donchian_channel_hband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband2(self):
        target = 'lower_band'
        result = donchian_channel_lband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband2(self):
        target = 'dc_band_width'
        result = donchian_channel_wband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband2(self):
        target = 'dc_percentage'
        result = donchian_channel_pband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestDonchianChannel2(unittest.TestCase):
    """
    https://www.investopedia.com/terms/d/donchianchannels.asp
    https://docs.google.com/spreadsheets/d/17JWWsxSiAb24BLzncUpccc8hg-03QjVWVXmoRCJ2lME/edit#gid=0
    """

    _filename = 'ta/tests/data/cs-dc2.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(
            high=cls._df['high'], low=cls._df['low'], close=cls._df['close'], n=20, offset=1, fillna=False)
        cls._indicator = DonchianChannel(**cls._params)

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

    def test_mavg2(self):
        target = 'middle_band'
        result = donchian_channel_mband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband2(self):
        target = 'upper_band'
        result = donchian_channel_hband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband2(self):
        target = 'lower_band'
        result = donchian_channel_lband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband2(self):
        target = 'dc_band_width'
        result = donchian_channel_wband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband2(self):
        target = 'dc_percentage'
        result = donchian_channel_pband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestKeltnerChannel(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    https://docs.google.com/spreadsheets/d/1qT8JbJ7F13bMV9-TcK-oFHL1F5sKPwakQWf6KrvGI3U/edit?usp=sharing
    """

    _filename = 'ta/tests/data/cs-kc.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(
            high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=20, n_atr=10, fillna=False, ov=False)
        cls._indicator = KeltnerChannel(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_mavg(self):
        target = 'middle_band'
        result = self._indicator.keltner_channel_mband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband(self):
        target = 'upper_band'
        result = self._indicator.keltner_channel_hband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband(self):
        target = 'lower_band'
        result = self._indicator.keltner_channel_lband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband(self):
        target = 'kc_band_width'
        result = self._indicator.keltner_channel_wband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband(self):
        target = 'kc_percentage'
        result = self._indicator.keltner_channel_pband()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator(self):
        target = 'kc_high_indicator'
        result = self._indicator.keltner_channel_hband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator(self):
        target = 'kc_low_indicator'
        result = self._indicator.keltner_channel_lband_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_mavg2(self):
        target = 'middle_band'
        result = keltner_channel_mband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband2(self):
        target = 'upper_band'
        result = keltner_channel_hband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband2(self):
        target = 'lower_band'
        result = keltner_channel_lband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wband2(self):
        target = 'kc_band_width'
        result = keltner_channel_wband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_pband2(self):
        target = 'kc_percentage'
        result = keltner_channel_pband(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_hband_indicator2(self):
        target = 'kc_high_indicator'
        result = keltner_channel_hband_indicator(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_lband_indicator2(self):
        target = 'kc_low_indicator'
        result = keltner_channel_lband_indicator(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
