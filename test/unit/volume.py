import unittest

import pandas as pd

from ta.volume import (
    AccDistIndexIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice,
    VolumePriceTrendIndicator,
    acc_dist_index,
    ease_of_movement,
    force_index,
    money_flow_index,
    on_balance_volume,
    sma_ease_of_movement,
    volume_weighted_average_price,
    volume_price_trend
)


class TestOnBalanceVolumeIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    """

    _filename = "test/data/cs-obv.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            close=cls._df["Close"], volume=cls._df["Volume"], fillna=False
        )
        cls._indicator = OnBalanceVolumeIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_obv(self):
        target = "OBV"
        result = on_balance_volume(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_obv2(self):
        target = "OBV"
        result = self._indicator.on_balance_volume()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )


class TestForceIndexIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:force_index
    """

    _filename = "test/data/cs-fi.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            close=cls._df["Close"], volume=cls._df["Volume"], window=13, fillna=False
        )
        cls._indicator = ForceIndexIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_fi(self):
        target = "FI"
        result = force_index(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_fi2(self):
        target = "FI"
        result = self._indicator.force_index()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )


class TestEaseOfMovementIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv
    """

    _filename = "test/data/cs-easeofmovement.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            high=cls._df["High"],
            low=cls._df["Low"],
            volume=cls._df["Volume"],
            window=14,
            fillna=False,
        )
        cls._indicator = EaseOfMovementIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_ease_of_movement(self):
        target = "EMV"
        result = ease_of_movement(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_ease_of_movement2(self):
        target = "EMV"
        result = self._indicator.ease_of_movement()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_sma_ease_of_movement(self):
        target = "SMA_EMV"
        result = sma_ease_of_movement(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_sma_ease_of_movement2(self):
        target = "SMA_EMV"
        result = self._indicator.sma_ease_of_movement()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )


class TestAccDistIndexIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    """

    _filename = "test/data/cs-accum.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            high=cls._df["High"],
            low=cls._df["Low"],
            close=cls._df["Close"],
            volume=cls._df["Volume"],
            fillna=False,
        )
        cls._indicator = AccDistIndexIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_adl(self):
        target = "ADLine"
        result = acc_dist_index(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_adl2(self):
        target = "ADLine"
        result = self._indicator.acc_dist_index()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )


class TestMFIIndicator(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:money_flow_index_mfi
    """

    _filename = "test/data/cs-mfi.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            high=cls._df["High"],
            low=cls._df["Low"],
            close=cls._df["Close"],
            volume=cls._df["Volume"],
            window=14,
            fillna=False,
        )
        cls._indicator = MFIIndicator(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_mfi(self):
        target = "MFI"
        result = self._indicator.money_flow_index()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_mfi2(self):
        target = "MFI"
        result = money_flow_index(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )


class TestVolumeWeightedAveragePrice(unittest.TestCase):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:vwap_intraday
    """

    _filename = "test/data/cs-vwap.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict(
            high=cls._df["High"],
            low=cls._df["Low"],
            close=cls._df["Close"],
            volume=cls._df["Volume"],
            fillna=False,
        )
        cls._indicator = VolumeWeightedAveragePrice(**cls._params)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_vwap(self):
        target = "vwap"
        result = volume_weighted_average_price(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_vwap2(self):
        target = "vwap"
        result = self._indicator.volume_weighted_average_price()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

class TestVolumePriceTrendIndicator(unittest.TestCase):
    """
    Original VPT: https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend
    One more: https://www.barchart.com/education/technical-indicators/price_volume_trend
    According to TradingView: PVT = [((CurrentClose - PreviousClose) / PreviousClose) x Volume] + PreviousPVT
    
    Smoothed version (by Alex Orekhov (everget)): https://ru.tradingview.com/script/3Ah2ALck-Price-Volume-Trend/
    His script is using `pvt` (TradingView built-in variable) as described in TradingView documentation of PVT and just smoothing it with ema or sma by choice.
    You can find smoothing here (13 row of script):
    `signal = signalType == "EMA" ? ema(pvt, signalLength) : sma(pvt, signalLength)`
    """

    _filename = "test/data/cs-vpt.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")
        cls._params = dict( # default VPT params, unsmoothed
            close=cls._df["Close"],
            volume=cls._df["Volume"],
            fillna=False,
            smoothing_factor=None,
            dropnans=False
        )
        cls._params_smoothed = dict( # smoothed VPT params
            close=cls._df["Close"],
            volume=cls._df["Volume"],
            fillna=False,
            smoothing_factor=14,
            dropnans=False
        )
        cls._indicator_default = VolumePriceTrendIndicator(**cls._params)
        cls._indicator_smoothed = VolumePriceTrendIndicator(**cls._params_smoothed)

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_vpt1(self):
        target = "unsmoothed vpt"
        result = volume_price_trend(**self._params)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_vpt2(self):
        target = "unsmoothed vpt"
        result = self._indicator_default.volume_price_trend()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_vpt3(self):
        target = "14-smoothed vpt"
        result = volume_price_trend(**self._params_smoothed)
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

    def test_vpt4(self):
        target = "14-smoothed vpt"
        result = self._indicator_default.volume_price_trend()
        pd.testing.assert_series_equal(
            self._df[target].tail(), result.tail(), check_names=False
        )

if __name__ == "__main__":
    unittest.main()
