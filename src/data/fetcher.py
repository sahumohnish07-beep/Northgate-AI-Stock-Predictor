"""Stock data ingestion module using yfinance with caching and validation."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yfinance as yf

from src.config import CACHE_DIR, DEFAULT_INTERVAL, DEFAULT_PERIOD
from src.utils.logger import get_logger

logger = get_logger("StockDataFetcher")


class StockDataFetcher:
    """Fetches, validates, caches, and pre-processes financial time-series data.

    Attributes:
        cache_dir: Directory where downloaded stock data files are cached.
        cache_ttl_hours: Time-to-live for cache in hours before re-fetching fresh data.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 12,
    ) -> None:
        """Initializes the StockDataFetcher.

        Args:
            cache_dir: Custom directory path for caching data. Defaults to config.CACHE_DIR.
            cache_ttl_hours: Number of hours after which cached data is refreshed.
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Builds a deterministic file path for caching.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            interval: Bar interval (e.g., '1d').

        Returns:
            Path to the cached CSV file.
        """
        clean_symbol = symbol.replace("^", "").replace("=", "").strip().upper()
        return self.cache_dir / f"{clean_symbol}_{interval}.parquet"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Determines if an existing cache file is fresh enough to use.

        Args:
            cache_file: Path to candidate cache file.

        Returns:
            True if cache exists and is newer than cache_ttl_hours; False otherwise.
        """
        if not cache_file.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=self.cache_ttl_hours)

    def fetch_history(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetches historical OHLCV data for a single ticker.

        Args:
            symbol: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA').
            start: Start date string (YYYY-MM-DD). If provided, overrides `period`.
            end: End date string (YYYY-MM-DD).
            period: Fallback data period if start/end omitted ('1mo', '1y', '3y', '5y', 'max').
            interval: Bar aggregation ('1d', '1wk', '1mo').
            use_cache: Whether to read/save to parquet cache.
            force_refresh: If True, ignores cache and downloads fresh data from Yahoo Finance.

        Returns:
            A clean, chronologically sorted pd.DataFrame with DateTime index and standardized columns:
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'Log_Return'].

        Raises:
            ValueError: If symbol is empty or no data could be retrieved.
        """
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            raise ValueError("Stock symbol cannot be empty.")

        cache_path = self._get_cache_path(clean_symbol, interval)

        # Check local cache first
        if use_cache and not force_refresh and self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {clean_symbol} from cache ({len(df)} records).")
                # Filter by start/end if specified
                if start:
                    df = df[df.index >= pd.to_datetime(start)]
                if end:
                    df = df[df.index <= pd.to_datetime(end)]
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {clean_symbol}: {e}. Re-downloading.")

        logger.info(
            f"Downloading {clean_symbol} [period={period}, start={start}, end={end}, interval={interval}]"
        )

        ticker = yf.Ticker(clean_symbol)
        if start:
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        else:
            df = ticker.history(period=period, interval=interval, auto_adjust=False)

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for symbol '{clean_symbol}'. "
                f"Verify the symbol exists and is supported."
            )

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is standard timezone-naive DatetimeIndex
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Sort chronologically ascending (strictly required for time series)
        df = df.sort_index(ascending=True)

        # Standardize required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Downloaded data missing required column '{col}'.")

        # Include Adj Close if available, else copy Close
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Drop rows with NaN in Close
        df = df.dropna(subset=["Close"]).copy()

        # Calculate standard price returns
        df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)
        import numpy as np
        # Log return = ln(P_t / P_{t-1})
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

        # Save to parquet cache
        if use_cache:
            try:
                df.to_parquet(cache_path)
                logger.info(f"Cached {clean_symbol} to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to write cache for {clean_symbol}: {e}")

        # Filter by start/end if specified
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetches historical data for multiple tickers.

        Args:
            symbols: List of ticker symbols.
            start: Optional start date string.
            end: Optional end date string.
            period: Fallback period.
            interval: Bar interval.
            use_cache: Whether to use caching.

        Returns:
            Dict mapping ticker symbol to its respective DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol.upper()] = self.fetch_history(
                    symbol=symbol,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        return results

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Fetches overview and fundamental metadata for a stock ticker.

        Args:
            symbol: Ticker symbol.

        Returns:
            Dictionary containing key financial metrics and company information.
        """
        clean_symbol = symbol.strip().upper()
        ticker = yf.Ticker(clean_symbol)
        try:
            info = ticker.info or {}
            return {
                "symbol": clean_symbol,
                "shortName": info.get("shortName", clean_symbol),
                "longName": info.get("longName", clean_symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", None),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", None)),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", None),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", None),
                "trailingPE": info.get("trailingPE", None),
                "forwardPE": info.get("forwardPE", None),
                "dividendYield": info.get("dividendYield", None),
                "currency": info.get("currency", "USD"),
                "summary": info.get("longBusinessSummary", ""),
            }
        except Exception as e:
            logger.warning(f"Could not retrieve extended info for {clean_symbol}: {e}")
            return {"symbol": clean_symbol, "shortName": clean_symbol}
