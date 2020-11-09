import unittest

import pandas as pd

from ta.momentum import (KAMAIndicator, PercentagePriceOscillator,
                         PercentageVolumeOscillator, ROCIndicator,
                         RSIIndicator, StochasticOscillator, StochRSIIndicator,
                         TSIIndicator, UltimateOscillator, WilliamsRIndicator,
                         kama, ppo, ppo_hist, ppo_signal, pvo, pvo_hist,
                         pvo_signal, roc, rsi, stoch, stoch_signal, stochrsi,
                         tsi, uo, wr)


class TestRateOfChangeIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    """

    _filename = 'ta/tests/data/cs-roc.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n=12, fillna=False)
        cls._indicator = ROCIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_roc(self):
        target = 'ROC'
        result = roc(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_roc2(self):
        target = 'ROC'
        result = self._indicator.roc()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestRSIIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Note: Using a more simple initilization (directly `ewm`; stockcharts uses `sma` + `ewm`)
    """

    _filename = 'ta/tests/data/cs-rsi.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n=14, fillna=False)
        cls._indicator = RSIIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_rsi(self):
        target = 'RSI'
        result = self._indicator.rsi()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_rsi2(self):
        target = 'RSI'
        result = rsi(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestStochRSIIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi
    """

    _filename = 'ta/tests/data/cs-stochrsi.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n=14, d1=3, d2=3, fillna=False)
        cls._indicator = StochRSIIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_stochrsi(self):
        target = 'StochRSI(14)'
        result = self._indicator.stochrsi()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_stochrsi2(self):
        target = 'StochRSI(14)'
        result = stochrsi(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestUltimateOscillator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:ultimate_oscillator
    """

    _filename = 'ta/tests/data/cs-ultosc.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(
            high=cls._df['High'],
            low=cls._df['Low'],
            close=cls._df['Close'],
            s=7,
            m=14,
            len=28,
            ws=4.0,
            wm=2.0,
            wl=1.0,
            fillna=False
        )
        cls._indicator = UltimateOscillator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_uo(self):
        target = 'Ult_Osc'
        result = self._indicator.uo()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_uo2(self):
        target = 'Ult_Osc'
        result = uo(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestStochasticOscillator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full
    """

    _filename = 'ta/tests/data/cs-soo.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(
            high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], n=14, d_n=3, fillna=False)
        cls._indicator = StochasticOscillator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_so(self):
        target = 'SO'
        result = self._indicator.stoch()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_so_signal(self):
        target = 'SO_SIG'
        result = self._indicator.stoch_signal()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_so2(self):
        target = 'SO'
        result = stoch(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_so_signal2(self):
        target = 'SO_SIG'
        result = stoch_signal(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestWilliamsRIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
    """

    _filename = 'ta/tests/data/cs-percentr.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(high=cls._df['High'], low=cls._df['Low'], close=cls._df['Close'], lbp=14, fillna=False)
        cls._indicator = WilliamsRIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_wr(self):
        target = 'Williams_%R'
        result = self._indicator.wr()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_wr2(self):
        target = 'Williams_%R'
        result = wr(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestKAMAIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
    """

    _filename = 'ta/tests/data/cs-kama.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n=10, pow1=2, pow2=30, fillna=False)
        cls._indicator = KAMAIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_kama(self):
        target = 'KAMA'
        result = self._indicator.kama()
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)

    def test_kama2(self):
        target = 'KAMA'
        result = kama(**self._params)
        pd.testing.assert_series_equal(self._df[target].tail(), result.tail(), check_names=False)


class TestTSIIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    """

    _filename = 'ta/tests/data/cs-tsi.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], r=25, s=13, fillna=False)
        cls._indicator = TSIIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_tsi(self):
        target = 'TSI'
        result = self._indicator.tsi()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False, check_less_precise=True)

    def test_tsi2(self):
        target = 'TSI'
        result = tsi(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False, check_less_precise=True)


class TestPercentagePriceOscillator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    https://docs.google.com/spreadsheets/d/1h9p8_PXU7G8sD-LciydpmH6rveaLwvoL7SMBGmO3kM4/edit#gid=0
    """

    _filename = 'ta/tests/data/cs-ppo.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(close=cls._df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=True)
        cls._indicator = PercentagePriceOscillator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_ppo(self):
        target = 'PPO'
        result = self._indicator.ppo()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_ppo2(self):
        target = 'PPO'
        result = ppo(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_ppo_signal(self):
        target = 'PPO_Signal_Line'
        result = self._indicator.ppo_signal()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_ppo_signal2(self):
        target = 'PPO_Signal_Line'
        result = ppo_signal(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_ppo_hist(self):
        target = 'PPO_Histogram'
        result = self._indicator.ppo_hist()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_ppo_hist2(self):
        target = 'PPO_Histogram'
        result = ppo_hist(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)


class TestPercentageVolumeOscillator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo
    https://docs.google.com/spreadsheets/d/1SyePHvrVBAcmjDiXe877Qrycx6TmajyrZ8UdrwVk9MI/edit#gid=0
    """

    _filename = 'ta/tests/data/cs-pvo.csv'

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=',')
        cls._params = dict(volume=cls._df['Volume'], n_slow=26, n_fast=12, n_sign=9, fillna=True)
        cls._indicator = PercentageVolumeOscillator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del(cls._df)

    def test_pvo(self):
        target = 'PVO'
        result = self._indicator.pvo()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_pvo2(self):
        target = 'PVO'
        result = pvo(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_pvo_signal(self):
        target = 'PVO_Signal_Line'
        result = self._indicator.pvo_signal()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_pvo_signal2(self):
        target = 'PVO_Signal_Line'
        result = pvo_signal(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_pvo_hist(self):
        target = 'PVO_Histogram'
        result = self._indicator.pvo_hist()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)

    def test_pvo_hist2(self):
        target = 'PVO_Histogram'
        result = pvo_hist(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False)
