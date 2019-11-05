import pandas as pd

from ta.tests.utils import TestIndicator
from ta.trend import CCIIndicator, ADXIndicator, VortexIndicator
from ta.trend import adx, adx_pos, adx_neg, cci, vortex_indicator_pos, vortex_indicator_neg


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
        result = ADXIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).adx()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos(self):
        target = '+DI14'
        result = adx_pos(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_pos2(self):
        target = '+DI14'
        result = ADXIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).adx_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg(self):
        target = '-DI14'
        result = adx_neg(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_adx_neg2(self):
        target = '-DI14'
        result = ADXIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).adx_neg()
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
        result = CCIIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=20, c=0.015, fillna=False).cci()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestVortexIndicator(TestIndicator):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """

    _filename = 'ta/tests/data/cs-vortex.csv'

    def test_vortex_indicator_pos(self):
        target = '+VI14'
        result = vortex_indicator_pos(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_pos2(self):
        target = '+VI14'
        result = VortexIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).vortex_indicator_pos()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg(self):
        target = '-VI14'
        result = vortex_indicator_neg(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_vortex_indicator_neg2(self):
        target = '-VI14'
        result = VortexIndicator(high=self._df['High'], low=self._df['Low'], close=self._df['Close'], n=14, fillna=False).vortex_indicator_neg()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)
