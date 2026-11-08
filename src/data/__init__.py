"""Data ingestion, pre-processing, and chronological time-series splitting."""

from src.data.fetcher import StockDataFetcher
from src.data.splitter import TimeSeriesSplitter

__all__ = ["StockDataFetcher", "TimeSeriesSplitter"]
