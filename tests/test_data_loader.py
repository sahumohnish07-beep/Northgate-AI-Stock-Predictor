"""Unit tests for Phase 1 Data Loader."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.data_loader import (
    DEFAULT_TICKERS,
    download_ticker_data,
    load_raw_data,
)


class TestDataLoader(unittest.TestCase):
    """Test suite verifying historical stock data downloading and parsing."""

    def setUp(self):
        """Create a temporary directory for test file downloads."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary test files."""
        self.temp_dir.cleanup()

    def test_download_single_ticker(self):
        """Verify that downloading a ticker saves a valid CSV with correct schema."""
        test_ticker = "ITC.NS"
        df = download_ticker_data(
            ticker=test_ticker,
            period="1mo",
            interval="1d",
            output_dir=self.output_dir,
        )

        self.assertIsNotNone(df, "Downloaded DataFrame should not be None.")
        self.assertFalse(df.empty, "DataFrame should contain records.")
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index must be DatetimeIndex.")
        self.assertTrue(df.index.is_monotonic_increasing, "Data must be sorted chronologically.")

        # Check required columns
        for col in ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]:
            self.assertIn(col, df.columns, f"Expected column '{col}' missing.")

        # Verify physical file existence
        target_csv = self.output_dir / f"{test_ticker}.csv"
        self.assertTrue(target_csv.exists(), f"File {target_csv} was not saved on disk.")

        # Verify load_raw_data loads the saved CSV
        loaded_df = load_raw_data(ticker=test_ticker, input_dir=self.output_dir)
        self.assertEqual(len(df), len(loaded_df), "Loaded rows should match downloaded rows.")
        pd.testing.assert_frame_equal(df, loaded_df, check_index_type=False)

    def test_default_tickers_list(self):
        """Verify that default Indian stock tickers are specified."""
        expected = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
        self.assertEqual(DEFAULT_TICKERS, expected)

    def test_invalid_ticker_handling(self):
        """Verify that an invalid ticker is handled gracefully without crashing."""
        df = download_ticker_data(
            ticker="INVALID_TICKER_XYZ_999",
            period="1mo",
            interval="1d",
            output_dir=self.output_dir,
        )
        self.assertIsNone(df, "Invalid ticker should return None.")


if __name__ == "__main__":
    unittest.main()
