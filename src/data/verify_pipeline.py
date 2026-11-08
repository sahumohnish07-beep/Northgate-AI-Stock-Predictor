"""CLI verification script for Phase 1 Data Ingestion & Chronological Splitting."""

import sys
from pathlib import Path

# Ensure root directory is on sys.path for direct script invocation
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.fetcher import StockDataFetcher
from src.data.splitter import TimeSeriesSplitter
from src.utils.logger import get_logger

logger = get_logger("VerifyPipeline")


def run_verification(symbol: str = "AAPL") -> None:
    """Executes an end-to-end test of the Phase 1 data pipeline.

    Args:
        symbol: Ticker symbol to test.
    """
    logger.info("=" * 60)
    logger.info("Northgate AI Stock Predictor - Phase 1 Verification")
    logger.info("=" * 60)

    fetcher = StockDataFetcher()

    logger.info(f"1. Fetching historical data for {symbol}...")
    df = fetcher.fetch_history(symbol=symbol, period="2y", interval="1d")
    logger.info(f"   Successfully fetched {len(df)} trading days.")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    logger.info(f"   Latest Close: ${df['Close'].iloc[-1]:.2f}")
    logger.info(f"   Latest Daily Return: {df['Daily_Return'].iloc[-1]*100:.2f}%")

    logger.info("2. Fetching company metadata...")
    info = fetcher.get_stock_info(symbol)
    logger.info(f"   Company: {info.get('longName', 'N/A')}")
    logger.info(f"   Sector:  {info.get('sector', 'N/A')}")
    logger.info(f"   52-Wk Range: ${info.get('fiftyTwoWeekLow', 0)} - ${info.get('fiftyTwoWeekHigh', 0)}")

    logger.info("3. Chronological Time-Series Splitting (Rule 7: Strict time order, no random shuffle)...")
    split = TimeSeriesSplitter.train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print("\n" + split.summary() + "\n")

    # Double check strict temporal sequence
    assert split.train.index.max() < split.val.index.min(), "Leakage between Train and Val"
    assert split.val.index.max() < split.test.index.min(), "Leakage between Val and Test"
    logger.info("   [PASS] Zero temporal leakage assertion verified!")

    logger.info("4. Testing walk-forward cross-validation folds...")
    folds = list(TimeSeriesSplitter.walk_forward_splits(df, n_splits=3, test_size=30))
    for i, (train_fold, test_fold) in enumerate(folds):
        logger.info(
            f"   Fold {i+1}: Train={len(train_fold)} rows [{train_fold.index.min().strftime('%Y-%m-%d')} to {train_fold.index.max().strftime('%Y-%m-%d')}], "
            f"Test={len(test_fold)} rows [{test_fold.index.min().strftime('%Y-%m-%d')} to {test_fold.index.max().strftime('%Y-%m-%d')}]"
        )

    logger.info("=" * 60)
    logger.info("Phase 1 Verification Succeeded! All checks passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_verification(test_symbol)
