"""LSTM Deep Learning model for stock price forecasting.

Northgate AI Stock Predictor - Phase 3.
Implements a sequential LSTM network using PyTorch to forecast next-day close
using a 60-day historical window. Fits MinMaxScaler strictly on train data,
includes EarlyStopping and ModelCheckpoint, evaluates with RMSE, MAE, MAPE,
plots predictions, and saves artifacts to models/ and data/processed/.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import pickle
import sys

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import DEFAULT_TICKERS, load_raw_data

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lstm_model")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class StockLSTM(nn.Module):
    """Deep Long Short-Term Memory (LSTM) network for stock price forecasting.

    Attributes:
        lstm: Stacked LSTM recurrent layers.
        fc: Fully connected projection layers to output next-day price.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initializes the StockLSTM network.

        Args:
            input_dim: Dimensionality of inputs per time step (1 for Close price).
            hidden_dim: Number of features in LSTM hidden states.
            num_layers: Number of recurrent layers.
            dropout: Dropout probability between recurrent layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM and feed-forward layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction


class EarlyStoppingAndCheckpoint:
    """Early stopping callback that halts training when validation loss stalls

    and checkpoints the model parameters with the best loss.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        patience: int = 10,
        min_delta: float = 1e-5,
    ) -> None:
        """Initializes the EarlyStopping and ModelCheckpoint handler.

        Args:
            checkpoint_path: Path where best model weights are saved.
            patience: Number of epochs to tolerate without validation improvement.
            min_delta: Minimum loss decrease to qualify as an improvement.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Evaluates validation loss at the end of an epoch.

        Args:
            val_loss: Current validation loss.
            model: Current PyTorch model instance.

        Returns:
            True if a new best model checkpoint was saved; False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(self.best_state_dict, self.checkpoint_path)
            logger.info(f"ModelCheckpoint: Saved new best model to {self.checkpoint_path} (val_loss: {val_loss:.6f})")
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"EarlyStopping: Triggered after {self.patience} epochs without improvement.")
            return False

    def restore_best(self, model: nn.Module) -> None:
        """Restores the best weights into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info("Restored model weights from best checkpoint.")


def prepare_lstm_data(
    df: pd.DataFrame,
    window_size: int = 60,
    train_ratio: float = 0.80,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    pd.DatetimeIndex,
]:
    """Prepares sequential data using a 60-day sliding window and MinMaxScaler fit on train only.

    Conforms to Rule 7: Time-series data must be split chronologically, never randomly.

    Args:
        df: Input DataFrame containing 'Close' price and DatetimeIndex.
        window_size: Lookback sequence length (60 days).
        train_ratio: Proportion of earliest historical data for training (0.80).

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler, test_dates).
    """
    df_sorted = df.sort_index(ascending=True)
    prices = df_sorted["Close"].values.reshape(-1, 1)

    n_samples = len(prices)
    split_idx = int(n_samples * train_ratio)

    train_raw = prices[:split_idx]
    test_raw = prices[split_idx:]

    # Scale with MinMaxScaler FIT ON TRAIN ONLY (strictly zero data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)

    # For testing, prepend the last `window_size` days from train so day 1 of test has a full 60-day input
    test_combined = np.concatenate([train_raw[-window_size:], test_raw], axis=0)
    test_scaled = scaler.transform(test_combined)

    # Build sliding sequences: X[i] = [p_{i}, ..., p_{i+59}], y[i] = p_{i+60}
    X_train_list, y_train_list = [], []
    for i in range(len(train_scaled) - window_size):
        X_train_list.append(train_scaled[i : i + window_size])
        y_train_list.append(train_scaled[i + window_size])

    X_test_list, y_test_list = [], []
    for i in range(len(test_scaled) - window_size):
        X_test_list.append(test_scaled[i : i + window_size])
        y_test_list.append(test_scaled[i + window_size])

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    test_dates = df_sorted.index[split_idx:]

    logger.info(
        f"Prepared 60-day LSTM sequences:\n"
        f"  X_train: {X_train.shape}, y_train: {y_train.shape}\n"
        f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}\n"
        f"  Test dates: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}"
    )

    return X_train, y_train, X_test, y_test, scaler, test_dates


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> StockLSTM:
    """Trains the StockLSTM model using Adam optimizer, MSE loss, EarlyStopping and ModelCheckpoint.

    Args:
        X_train: Training sequences of shape (N, 60, 1).
        y_train: Training targets of shape (N, 1).
        X_val: Validation sequences.
        y_val: Validation targets.
        checkpoint_path: Destination path for best model checkpoint.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate for Adam.
        patience: Early stopping patience.
        device: Torch execution device (cpu or cuda).

    Returns:
        Trained StockLSTM with restored best checkpoint weights.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training LSTM on device: {dev}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StockLSTM(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStoppingAndCheckpoint(checkpoint_path=checkpoint_path, patience=patience)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(val_dataset)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # EarlyStopping & ModelCheckpoint step
        early_stopping.step(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping halting training at epoch {epoch}.")
            break

    early_stopping.restore_best(model)
    return model


def evaluate_lstm(
    model: StockLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Generates predictions and evaluates RMSE, MAE, and MAPE on the original price scale (INR).

    Args:
        model: Trained StockLSTM model.
        X_test: Test sequences.
        y_test: Test targets (scaled).
        scaler: MinMaxScaler fitted on training data.
        device: Torch execution device.

    Returns:
        Tuple of (metrics_dict, actual_prices_inr, predicted_prices_inr).
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(dev)
        scaled_preds = model(test_tensor).cpu().numpy()

    # Inverse transform to original currency scale (INR)
    pred_inr = scaler.inverse_transform(scaled_preds).flatten()
    actual_inr = scaler.inverse_transform(y_test).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual_inr, pred_inr)))
    mae = float(mean_absolute_error(actual_inr, pred_inr))
    mape = float(np.mean(np.abs((actual_inr - pred_inr) / actual_inr)) * 100.0)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_%": round(mape, 4),
    }

    return metrics, actual_inr, pred_inr


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: pd.DatetimeIndex,
    ticker: str,
    output_path: Path,
) -> None:
    """Plots actual vs predicted next-day close prices and saves high-resolution figure.

    Args:
        actual: Ground truth closing prices.
        predicted: Predicted closing prices.
        dates: DatetimeIndex corresponding to test samples.
        ticker: Ticker symbol.
        output_path: File path to save the chart.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label=f"Actual {ticker} Close", color="#1f77b4", linewidth=2.0)
    plt.plot(dates, predicted, label=f"LSTM Predicted Close (60d window)", color="#ff7f0e", linestyle="--", linewidth=2.0)
    plt.title(f"{ticker} - Actual vs LSTM Predicted Next-Day Close Price", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (INR)", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved prediction plot to {output_path}")


def run_lstm_pipeline(
    ticker: str = "RELIANCE.NS",
    window_size: int = 60,
    train_ratio: float = 0.80,
    epochs: int = 50,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Executes the full Phase 3 LSTM pipeline for a stock.

    Args:
        ticker: Target stock symbol (default 'RELIANCE.NS').
        window_size: Sequence window length (60 days).
        train_ratio: 80% train split.
        epochs: Training epochs.

    Returns:
        Tuple of (lstm_metrics, comparison_dataframe).
    """
    logger.info("=" * 70)
    logger.info(f"Phase 3: Training 60-day LSTM Model on {ticker}")
    logger.info("=" * 70)

    # 1. Load data
    df = load_raw_data(ticker)

    # 2. Prepare 60-day sequences and fit scaler on train only
    X_train, y_train, X_test, y_test, scaler, test_dates = prepare_lstm_data(
        df,
        window_size=window_size,
        train_ratio=train_ratio,
    )

    # Further partition train into train (85%) and val (15%) chronologically for EarlyStopping
    val_split = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:val_split], y_train[:val_split]
    X_val, y_val = X_train[val_split:], y_train[val_split:]

    # 3. Model Checkpoint & Early Stopping Setup
    checkpoint_file = MODELS_DIR / "lstm_model.pt"
    scaler_file = MODELS_DIR / "scaler.pkl"

    # Save fitted scaler as .pkl
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved MinMaxScaler to {scaler_file}")

    # 4. Train LSTM
    model = train_lstm(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        checkpoint_path=checkpoint_file,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.001,
        patience=10,
    )

    # 5. Evaluate on test set
    metrics, actual_inr, pred_inr = evaluate_lstm(model, X_test, y_test, scaler)
    logger.info(f"LSTM Test Performance on {ticker}:")
    logger.info(f"  RMSE:    {metrics['RMSE']:>7.2f}")
    logger.info(f"  MAE:     {metrics['MAE']:>7.2f}")
    logger.info(f"  MAPE:    {metrics['MAPE_%']:>5.2f}%")

    # 6. Plot actual vs predicted
    plot_file = PROCESSED_DIR / "lstm_actual_vs_predicted.png"
    plot_predictions(actual_inr, pred_inr, test_dates, ticker, plot_file)

    # 7. Compare against Phase 2 baselines
    baseline_csv = PROCESSED_DIR / "baseline_results.csv"
    comparison_rows = []

    if baseline_csv.exists():
        base_df = pd.read_csv(baseline_csv)
        ticker_base = base_df[base_df["Ticker"] == ticker].copy()
        for _, row in ticker_base.iterrows():
            comparison_rows.append({
                "Ticker": ticker,
                "Model": row["Model"],
                "Test_RMSE": row["Test_RMSE"],
                "Test_MAE": row["Test_MAE"],
                "Test_MAPE_%": row["Test_MAPE_%"],
            })

    comparison_rows.append({
        "Ticker": ticker,
        "Model": "LSTM (60-day window)",
        "Test_RMSE": metrics["RMSE"],
        "Test_MAE": metrics["MAE"],
        "Test_MAPE_%": metrics["MAPE_%"],
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = PROCESSED_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info(f"Saved comparison table against Phase 2 baselines to {comparison_csv}")

    return metrics, comparison_df


if __name__ == "__main__":
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    metrics_out, comp_table = run_lstm_pipeline(ticker=test_ticker)
    print("\n" + "=" * 70)
    print(f"NORTHGATE AI STOCK PREDICTOR - MODEL COMPARISON ON {test_ticker}")
    print("=" * 70)
    print(comp_table.to_string(index=False))
    print("=" * 70)

