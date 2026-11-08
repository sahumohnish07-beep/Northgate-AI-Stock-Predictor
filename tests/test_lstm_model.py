"""Unit tests for Phase 3: LSTM price prediction model."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.lstm_model import (
    EarlyStoppingAndCheckpoint,
    StockLSTM,
    evaluate_lstm,
    prepare_lstm_data,
)


class TestLSTMModel(unittest.TestCase):
    """Test suite for sequential dataset construction, LSTM forward pass, and checkpointing."""

    def setUp(self):
        """Create synthetic historical stock price series."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        close = 150.0 + np.cumsum(np.random.randn(200) * 2.0)
        self.df = pd.DataFrame({"Close": close}, index=dates)

    def test_prepare_lstm_data_shapes(self):
        """Verify 60-day sliding window sequences and leak-free scaling."""
        window_size = 60
        train_ratio = 0.80

        X_train, y_train, X_test, y_test, scaler, test_dates = prepare_lstm_data(
            self.df,
            window_size=window_size,
            train_ratio=train_ratio,
        )

        n_samples = len(self.df)
        n_train = int(n_samples * train_ratio)
        n_test = n_samples - n_train

        self.assertEqual(X_train.shape[1], window_size)
        self.assertEqual(X_train.shape[2], 1)
        self.assertEqual(len(X_train), n_train - window_size)
        self.assertEqual(len(X_test), n_test)
        self.assertEqual(len(test_dates), n_test)

        # Scaler must be fitted on train data values only
        train_min = self.df["Close"].iloc[:n_train].min()
        train_max = self.df["Close"].iloc[:n_train].max()
        self.assertAlmostEqual(scaler.data_min_[0], train_min, places=4)
        self.assertAlmostEqual(scaler.data_max_[0], train_max, places=4)

    def test_stock_lstm_forward(self):
        """Verify forward pass output dimensions for batch input."""
        batch_size = 16
        seq_len = 60
        input_dim = 1

        model = StockLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
        dummy_input = torch.randn(batch_size, seq_len, input_dim)
        output = model(dummy_input)

        self.assertEqual(output.shape, (batch_size, 1))

    def test_early_stopping_and_checkpoint(self):
        """Verify early stopping triggers after patience runs out and checkpoints are saved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"
            cb = EarlyStoppingAndCheckpoint(checkpoint_path=ckpt_path, patience=3)
            model = StockLSTM(input_dim=1, hidden_dim=16, num_layers=1)

            # Improving loss saves checkpoint
            saved = cb.step(0.10, model)
            self.assertTrue(saved)
            self.assertTrue(ckpt_path.exists())

            # Worse loss does not save and increments patience counter
            self.assertFalse(cb.step(0.12, model))
            self.assertEqual(cb.counter, 1)
            self.assertFalse(cb.early_stop)

            self.assertFalse(cb.step(0.11, model))
            self.assertEqual(cb.counter, 2)
            self.assertFalse(cb.early_stop)

            # Reaches patience limit
            self.assertFalse(cb.step(0.13, model))
            self.assertTrue(cb.early_stop)


if __name__ == "__main__":
    unittest.main()
