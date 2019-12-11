import unittest

import pandas as pd

from ta.tests.utils import TestIndicator
from ta.trend import (MACD, ADXIndicator, CCIIndicator, PSARIndicator,
                      VortexIndicator, adx, adx_neg, adx_pos, cci, macd,
                      macd_diff, macd_signal, psar_down, psar_down_indicator,
                      psar_up, psar_up_indicator, vortex_indicator_neg,
                      vortex_indicator_pos)


class TestADXIndicator(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx
    """

    _filename = 'ta/tests/data/cs-adx.csv'

    def test_adx(self):
        target = 'ADX'
        result = adx(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx2(self):
        target = 'ADX'
        result = ADXIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).adx()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos(self):
        target = '+DI14'
        result = adx_pos(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos2(self):
        target = '+DI14'
        result = ADXIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).adx_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg(self):
        target = '-DI14'
        result = adx_neg(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg2(self):
        target = '-DI14'
        result = ADXIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14,
            fillna=False).adx_neg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestMACDIndicator(TestIndicator):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
    """

    _filename = 'ta/tests/data/cs-macd.csv'

    def test_macd(self):
        target = 'MACD_line'
        result = macd(close=self._df['Close'], n_slow=26, n_fast=12, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd2(self):
        target = 'MACD_line'
        result = MACD(close=self._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False).macd()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_signal(self):
        target = 'MACD_signal'
        result = macd_signal(close=self._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_signal2(self):
        target = 'MACD_signal'
        result = MACD(close=self._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False).macd_signal()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_diff(self):
        target = 'MACD_diff'
        result = macd_diff(close=self._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_macd_diff2(self):
        target = 'MACD_diff'
        result = MACD(close=self._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False).macd_diff()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestCCIIndicator(TestIndicator):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    _filename = 'ta/tests/data/cs-cci.csv'

    def test_cci(self):
        target = 'CCI'
        result = cci(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=20, c=0.015, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_cci2(self):
        target = 'CCI'
        result = CCIIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=20, c=0.015,
            fillna=False).cci()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestVortexIndicator(TestIndicator):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    _filename = 'ta/tests/data/cs-vortex.csv'

    def test_vortex_indicator_pos(self):
        target = '+VI14'
        result = vortex_indicator_pos(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_pos2(self):
        target = '+VI14'
        result = VortexIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14,
            fillna=False).vortex_indicator_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg(self):
        target = '-VI14'
        result = vortex_indicator_neg(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg2(self):
        target = '-VI14'
        result = VortexIndicator(
            high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14,
            fillna=False).vortex_indicator_neg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestPSARIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
    """

    _filename = 'ta/tests/data/cs-psar.csv'

    def setUp(self):
        self._df = pd.read_csv(self._filename, sep=',')
        self._indicator = PSARIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'],
                                        fillna=False)

    def tearDown(self):
        del(self._df)

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
