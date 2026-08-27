import unittest

import pandas as pd

from ta.volatility import AverageTrueRange, average_true_range


class TestATRShortInput(unittest.TestCase):
    def test_short_inputs(self):
        for size in (0, 1, 5):
            for fillna in (False, True):
                with self.subTest(size=size, fillna=fillna):
                    index = pd.date_range("2021-01-01", periods=size)
                    close = pd.Series(range(2, size + 2), index=index, dtype=float)
                    kwargs = dict(
                        high=close + 1,
                        low=close - 1,
                        close=close,
                        window=6,
                        fillna=fillna,
                    )
                    expected = pd.Series(0.0, index=index, name="atr")
                    pd.testing.assert_series_equal(
                        average_true_range(**kwargs), expected
                    )
                    pd.testing.assert_series_equal(
                        AverageTrueRange(**kwargs).average_true_range(), expected
                    )

    def test_complete_window(self):
        for size in (6, 7):
            with self.subTest(size=size):
                index = pd.date_range("2021-01-01", periods=size)
                close = pd.Series(range(2, size + 2), index=index, dtype=float)
                expected = pd.Series(
                    [0.0] * 5 + [2.0] * (size - 5), index=index, name="atr"
                )
                result = average_true_range(close + 1, close - 1, close, window=6)
                pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
