import unittest

import pandas as pd

from ta.trend import (MACD, ADXIndicator, CCIIndicator, PSARIndicator,
                      VortexIndicator, adx, adx_neg, adx_pos, cci, macd,
                      macd_diff, macd_signal, psar_down, psar_down_indicator,
                      psar_up, psar_up_indicator, vortex_indicator_neg,
                      vortex_indicator_pos)


class TestADXIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx
    """

    _filename = 'ta/tests/data/cs-adx.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=14, fillna=False)
        cls._indicator = ADXIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_adx(self):
        target = 'ADX'
        result = adx(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx2(self):
        target = 'ADX'
        result = self._indicator.adx()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos(self):
        target = '+DI14'
        result = adx_pos(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos2(self):
        target = '+DI14'
        result = self._indicator.adx_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg(self):
        target = '-DI14'
        result = adx_neg(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg2(self):
        target = '-DI14'
        result = self._indicator.adx_neg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestMACDIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
    """

    _filename = 'ta/tests/data/cs-macd.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
        cls._indicator = MACD(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del (cls._df)

    def test_macd(self):
        target = 'MACD_line'
        result = macd(close=self._df['Close'], n_slow=26, n_fast=12, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd2(self):
        target = 'MACD_line'
        result = self._indicator.macd()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_signal(self):
        target = 'MACD_signal'
        result = macd_signal(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_signal2(self):
        target = 'MACD_signal'
        result = MACD(**self._params).macd_signal()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_diff(self):
        target = 'MACD_diff'
        result = macd_diff(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_diff2(self):
        target = 'MACD_diff'
        result = MACD(**self._params).macd_diff()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestCCIIndicator(unittest.TestCase):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    _filename = 'ta/tests/data/cs-cci.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(
            high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=20, c=0.015, fillna=False)
        cls._indicator = CCIIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del (cls._df)

    def test_cci(self):
        target = 'CCI'
        result = cci(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_cci2(self):
        target = 'CCI'
        result = self._indicator.cci()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestVortexIndicator(unittest.TestCase):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    _filename = 'ta/tests/data/cs-vortex.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=14, fillna=False)
        cls._indicator = VortexIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del (cls._df)

    def test_vortex_indicator_pos(self):
        target = '+VI14'
        result = vortex_indicator_pos(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_pos2(self):
        target = '+VI14'
        result = self._indicator.vortex_indicator_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg(self):
        target = '-VI14'
        result = vortex_indicator_neg(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg2(self):
        target = '-VI14'
        result = self._indicator.vortex_indicator_neg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestPSARIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
    """

    _filename = 'ta/tests/data/cs-psar.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], fillna=False)
        cls._indicator = PSARIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del (cls._df)

    def test_psar_up(self):
        target = 'psar_up'
        result = self._indicator.psar_up()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_down(self):
        target = 'psar_down'
        result = self._indicator.psar_down()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_up_indicator(self):
        target = 'psar_up_ind'
        result = self._indicator.psar_up_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_down_indicator(self):
        target = 'psar_down_ind'
        result = self._indicator.psar_down_indicator()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_up2(self):
        target = 'psar_up'
        result = psar_up(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_down2(self):
        target = 'psar_down'
        result = psar_down(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_up_indicator2(self):
        target = 'psar_up_ind'
        result = psar_up_indicator(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_psar_down_indicator2(self):
        target = 'psar_down_ind'
        result = psar_down_indicator(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
