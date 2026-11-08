"""Unit tests for Phase 1: Stock data fetching and chronological time-series splitting."""

import unittest
import pandas as pd
from src.data.fetcher import StockDataFetcher
from src.data.splitter import TimeSeriesSplitter


class TestDataPipeline(unittest.TestCase):
    """Test suite for the stock data pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once before running tests."""
        cls.fetcher = StockDataFetcher()
        cls.symbol = "AAPL"
        # Fetch 1 year of data for tests
        cls.df = cls.fetcher.fetch_history(symbol=cls.symbol, period="1y", interval="1d")

    def test_fetch_history_structure(self):
        """Verify that fetched historical data meets all schema and ordering requirements."""
        self.assertFalse(self.df.empty, "DataFrame should not be empty.")
        self.assertIsInstance(self.df.index, pd.DatetimeIndex, "Index must be a DatetimeIndex.")
        self.assertTrue(self.df.index.is_monotonic_increasing, "Index must be strictly chronologically increasing.")

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Daily_Return",
            "Log_Return",
        ]
        for col in required_columns:
            self.assertIn(col, self.df.columns, f"Missing expected column: {col}")

        # Returns should not have NaN values
        self.assertEqual(self.df["Daily_Return"].isna().sum(), 0, "Daily_Return should not have NaN.")
        self.assertEqual(self.df["Log_Return"].isna().sum(), 0, "Log_Return should not have NaN.")

    def test_cache_functionality(self):
        """Verify that fetching with use_cache=True re-uses cached data without errors."""
        cached_df = self.fetcher.fetch_history(symbol=self.symbol, period="1y", interval="1d", use_cache=True)
        self.assertEqual(len(self.df), len(cached_df), "Cached row count must match original row count.")
        pd.testing.assert_frame_equal(self.df, cached_df)

    def test_chronological_train_val_test_split(self):
        """Verify Rule 7: Strict chronological splitting without temporal leakage."""
        split_result = TimeSeriesSplitter.train_val_test_split(
            self.df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        train_df = split_result.train
        val_df = split_result.val
        test_df = split_result.test

        # Verify sizes
        total_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_rows, len(self.df), "Total partition rows must equal original DataFrame length.")

        # Verify strict time ordering (No lookahead bias)
        self.assertTrue(
            train_df.index.max() < val_df.index.min(),
            f"Leakage detected: Train max ({train_df.index.max()}) not before Val min ({val_df.index.min()})",
        )
        self.assertTrue(
            val_df.index.max() < test_df.index.min(),
            f"Leakage detected: Val max ({val_df.index.max()}) not before Test min ({test_df.index.min()})",
        )

    def test_walk_forward_splits(self):
        """Verify walk-forward time-series CV folds maintain chronological sequence."""
        folds = list(TimeSeriesSplitter.walk_forward_splits(self.df, n_splits=3, test_size=20))
        self.assertEqual(len(folds), 3, "Expected 3 walk-forward folds.")

        for i, (train_fold, test_fold) in enumerate(folds):
            self.assertEqual(len(test_fold), 20, f"Test fold {i} should contain 20 records.")
            self.assertTrue(
                train_fold.index.max() < test_fold.index.min(),
                f"Fold {i} has temporal leakage: train max is not < test min.",
            )

    def test_stock_info(self):
        """Verify fundamental metadata retrieval."""
        info = self.fetcher.get_stock_info(self.symbol)
        self.assertEqual(info.get("symbol"), "AAPL")
        self.assertIn("shortName", info)


if __name__ == "__main__":
    unittest.main()

