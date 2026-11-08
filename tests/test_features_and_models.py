"""Unit tests for Phase 2: Feature engineering and baseline ML models."""

import unittest
import numpy as np
import pandas as pd

from src.features import (
    compute_bollinger_bands,
    compute_macd,
    compute_moving_averages,
    compute_rsi,
    engineer_features,
)
from src.models import compute_metrics, split_time_series_80_20


class TestFeaturesAndModels(unittest.TestCase):
    """Test suite for feature computation, temporal splitting, and metric evaluation."""

    def setUp(self):
        """Create synthetic OHLCV time-series dataframe for unit testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        close = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
        high = close + np.random.uniform(0.5, 2.0, size=100)
        low = close - np.random.uniform(0.5, 2.0, size=100)
        open_price = low + np.random.uniform(0.1, 1.5, size=100)
        volume = np.random.randint(100000, 5000000, size=100)

        self.df = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    def test_compute_rsi(self):
        """Verify RSI calculation range and constraints."""
        rsi = compute_rsi(self.df["Close"], period=14)
        self.assertEqual(len(rsi), len(self.df))
        self.assertTrue((rsi >= 0.0).all() and (rsi <= 100.0).all(), "RSI values must stay between 0 and 100.")

    def test_compute_macd(self):
        """Verify MACD components and dimensions."""
        macd_df = compute_macd(self.df["Close"])
        for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
            self.assertIn(col, macd_df.columns)
        self.assertEqual(len(macd_df), len(self.df))

    def test_compute_bollinger_bands(self):
        """Verify Bollinger Bands ordering (Upper >= Middle >= Lower)."""
        bb_df = compute_bollinger_bands(self.df["Close"], period=20)
        valid_rows = bb_df.dropna()
        self.assertTrue((valid_rows["BB_Upper"] >= valid_rows["BB_Middle"]).all())
        self.assertTrue((valid_rows["BB_Middle"] >= valid_rows["BB_Lower"]).all())

    def test_compute_moving_averages(self):
        """Verify SMA and EMA calculation."""
        ma_df = compute_moving_averages(self.df["Close"], windows=[10, 20])
        self.assertIn("SMA_10", ma_df.columns)
        self.assertIn("EMA_10", ma_df.columns)
        self.assertIn("SMA_20", ma_df.columns)
        self.assertIn("EMA_20", ma_df.columns)

    def test_engineer_features_pipeline(self):
        """Verify feature engineering pipeline produces non-empty feature matrix without NaNs."""
        feat_df = engineer_features(self.df, include_target=True)
        self.assertFalse(feat_df.empty)
        self.assertEqual(feat_df.isna().sum().sum(), 0, "No NaNs allowed in final feature matrix.")
        self.assertIn("Target_Next_Close", feat_df.columns)
        self.assertIn("RSI_14", feat_df.columns)
        self.assertIn("MACD", feat_df.columns)
        self.assertIn("Close_Lag_1", feat_df.columns)

    def test_split_time_series_80_20(self):
        """Verify Rule 7: Strict chronological 80/20 train/test split without lookahead leakage."""
        feat_df = engineer_features(self.df, include_target=True)
        X_train, y_train, X_test, y_test, feature_cols = split_time_series_80_20(
            feat_df,
            target_col="Target_Next_Close",
            train_ratio=0.80,
        )

        self.assertEqual(len(X_train) + len(X_test), len(feat_df))
        self.assertEqual(len(X_train), int(len(feat_df) * 0.80))
        self.assertTrue(X_train.index.max() < X_test.index.min(), "Temporal leakage detected between train and test.")
        self.assertNotIn("Target_Next_Close", feature_cols)

    def test_compute_metrics(self):
        """Verify RMSE, MAE, and MAPE calculations."""
        y_true = np.array([100.0, 110.0, 120.0])
        y_pred = np.array([102.0, 108.0, 122.0])
        metrics = compute_metrics(y_true, y_pred)

        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("MAPE_%", metrics)
        self.assertAlmostEqual(metrics["MAE"], 2.0, places=2)
        self.assertGreater(metrics["RMSE"], 0.0)
        self.assertGreater(metrics["MAPE_%"], 0.0)


if __name__ == "__main__":
    unittest.main()

