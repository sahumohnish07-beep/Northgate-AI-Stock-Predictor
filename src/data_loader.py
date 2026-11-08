"""Data loader module for Northgate AI Stock Predictor.

Fetches 5 years of daily OHLCV historical market data for Indian equities
(RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ITC.NS) using yfinance, handles missing data,
and persists cleaned datasets to data/raw/<ticker>.csv.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_loader")

DEFAULT_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ITC.NS",
]
DEFAULT_PERIOD: str = "5y"
DEFAULT_INTERVAL: str = "1d"
DEFAULT_OUTPUT_DIR: Path = Path("data/raw")


def download_ticker_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
) -> Optional[pd.DataFrame]:
    """Downloads historical OHLCV data for a given ticker and saves to CSV.

    Args:
        ticker: Stock ticker symbol (e.g. 'RELIANCE.NS').
        period: Historical period to fetch (e.g. '5y').
        interval: Granularity of data (e.g. '1d').
        output_dir: Directory where raw CSV files are saved.

    Returns:
        pd.DataFrame containing clean OHLCV data, or None if download failed.
    """
    clean_ticker = ticker.strip().upper()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_file = out_path / f"{clean_ticker}.csv"

    logger.info(f"Downloading {clean_ticker} [period={period}, interval={interval}]...")

    try:
        yf_ticker = yf.Ticker(clean_ticker)
        df = yf_ticker.history(period=period, interval=interval, auto_adjust=False)

        if df is None or df.empty:
            logger.error(f"No data returned for ticker '{clean_ticker}'.")
            return None

        # Flatten multi-level columns if present (from recent yfinance changes)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize index to timezone-naive DatetimeIndex with uniform date format
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d"))
        df.index.name = "Date"

        # Ensure chronological ascending sort (Rule 7: Never random, strictly temporal)
        df = df.sort_index(ascending=True)

        # Check required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(
                    f"Missing required OHLCV column '{col}' for {clean_ticker}."
                )
                return None

        # Fallback for Adj Close if not provided
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Handle missing data: drop rows where Close is NaN, forward-fill non-zero values
        initial_len = len(df)
        df = df.dropna(subset=["Close"]).copy()
        df = df.ffill().bfill()

        # Remove trading days with 0 or negative prices
        df = df[(df["Close"] > 0) & (df["Open"] > 0)].copy()

        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            logger.warning(
                f"Handled missing data for {clean_ticker}: removed {dropped_rows} bad/empty rows."
            )

        # Compute basic return metric for convenience
        df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)

        # Save to CSV in data/raw/<ticker>.csv
        df.to_csv(target_file, index=True)
        logger.info(
            f"Successfully saved {clean_ticker} ({len(df)} trading days) to {target_file} "
            f"[{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}]."
        )
        return df

    except Exception as exc:
        logger.error(
            f"Failed to download data for ticker '{clean_ticker}': {exc}",
            exc_info=False,
        )
        return None


def download_multiple_tickers(
    tickers: Optional[List[str]] = None,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Downloads historical OHLCV data for multiple tickers.

    Args:
        tickers: List of ticker symbols. If None, defaults to DEFAULT_TICKERS.
        period: Period duration string (e.g. '5y').
        interval: Bar interval string (e.g. '1d').
        output_dir: Directory where CSVs are saved.

    Returns:
        Dict mapping ticker symbol to its downloaded pd.DataFrame.
    """
    tickers_to_fetch = tickers or DEFAULT_TICKERS
    downloaded_data: Dict[str, pd.DataFrame] = {}

    logger.info(
        f"Starting batch download for {len(tickers_to_fetch)} tickers: {tickers_to_fetch}"
    )

    for ticker in tickers_to_fetch:
        df = download_ticker_data(
            ticker, period=period, interval=interval, output_dir=output_dir
        )
        if df is not None:
            downloaded_data[ticker.upper()] = df
        else:
            logger.warning(f"Skipping failed ticker: {ticker}")

    logger.info(
        f"Batch download complete. Successfully downloaded {len(downloaded_data)}/{len(tickers_to_fetch)} tickers."
    )
    return downloaded_data


def load_raw_data(
    ticker: str, input_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR
) -> pd.DataFrame:
    """Loads a previously downloaded raw ticker CSV into a clean DataFrame.

    Args:
        ticker: Ticker symbol (e.g. 'RELIANCE.NS').
        input_dir: Directory containing raw CSV files.

    Returns:
        pd.DataFrame with DatetimeIndex and standard OHLCV columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    clean_ticker = ticker.strip().upper()
    file_path = Path(input_dir) / f"{clean_ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file for '{clean_ticker}' not found at {file_path}. "
            f"Run download_ticker_data('{clean_ticker}') first."
        )

    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df = df.sort_index(ascending=True)
    return df


if __name__ == "__main__":
    cli_tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    download_multiple_tickers(cli_tickers)
