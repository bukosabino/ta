import unittest

import pandas as pd

import ta


class TestWrapper(unittest.TestCase):

    _filename = "test/data/datas.csv"

    @classmethod
    def setUpClass(cls):
        cls._df = pd.read_csv(cls._filename, sep=",")

    @classmethod
    def tearDownClass(cls):
        del cls._df

    def test_general(self):
        # Clean nan values
        df = ta.utils.dropna(self._df)

        # Add all ta features filling nans values
        ta.add_all_ta_features(
            df=df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume_BTC",
            fillna=True,
        )

        # Add all ta features not filling nans values
        ta.add_all_ta_features(
            df=df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume_BTC",
            fillna=False,
        )

        # Check added ta features are all numerical values after filling nans
        input_cols = self._df.columns
        df_with_ta = ta.add_all_ta_features(
            df=df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume_BTC",
            fillna=True,
        )
        ta_cols = [c for c in df_with_ta.columns if c not in input_cols]
        self.assertTrue(
            df_with_ta[ta_cols]
            .apply(lambda series: pd.to_numeric(series, errors="coerce"))
            .notnull()
            .all()
            .all()
        )

        self.assertTrue(df_with_ta.shape[1] == 94)

    def test_only_vectorized(self):
        # Clean nan values
        df = ta.utils.dropna(self._df)

        # Add all ta features filling nans values
        df_vectorized = ta.add_all_ta_features(
            df=df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume_BTC",
            fillna=True,
            vectorized=True
        )

        self.assertTrue(df_vectorized.shape[1] == 76)


if __name__ == "__main__":
    unittest.main()
