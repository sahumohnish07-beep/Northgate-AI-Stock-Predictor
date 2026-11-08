"""Chronological time-series splitting utilities.

Enforces strict temporal partitioning with zero lookahead bias or random sampling.
Conforms to Rule 7: 'Time-series data must be split by time, never randomly.'
"""

from dataclasses import dataclass
from typing import Generator, Optional, Tuple
import pandas as pd

from src.config import (
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
)
from src.utils.logger import get_logger

logger = get_logger("TimeSeriesSplitter")


@dataclass(frozen=True)
class TimeSeriesSplitResult:
    """Holds partitioned datasets and temporal boundary metadata."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_range: Tuple[pd.Timestamp, pd.Timestamp]

    def summary(self) -> str:
        """Returns a formatted string summary of the partitions."""
        return (
            f"Train set: {len(self.train):>5} rows ({self.train_range[0].strftime('%Y-%m-%d')} to {self.train_range[1].strftime('%Y-%m-%d')})\n"
            f"Val set:   {len(self.val):>5} rows ({self.val_range[0].strftime('%Y-%m-%d')} to {self.val_range[1].strftime('%Y-%m-%d')})\n"
            f"Test set:  {len(self.test):>5} rows ({self.test_range[0].strftime('%Y-%m-%d')} to {self.test_range[1].strftime('%Y-%m-%d')})"
        )


class TimeSeriesSplitter:
    """Provides temporal splitting strategies for financial time series data."""

    @staticmethod
    def train_val_test_split(
        df: pd.DataFrame,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
    ) -> TimeSeriesSplitResult:
        """Splits a DataFrame chronologically into train, validation, and test sets.

        The DataFrame index must be a DatetimeIndex or sorted datetime values.
        Under no circumstances is random shuffling permitted.

        Args:
            df: Input DataFrame with DatetimeIndex.
            train_ratio: Proportion of data allocated to training (e.g. 0.70).
            val_ratio: Proportion of data allocated to validation (e.g. 0.15).
            test_ratio: Proportion of data allocated to testing (e.g. 0.15).

        Returns:
            TimeSeriesSplitResult containing the three slices and date metadata.

        Raises:
            ValueError: If ratios don't sum to ~1.0, or dataset has too few rows.
        """
        if df is None or len(df) < 10:
            raise ValueError(f"Insufficient data for time-series splitting: {len(df) if df is not None else 0} rows.")

        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0. Given: train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        # Ensure sorted chronologically
        df_sorted = df.sort_index(ascending=True)

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Ensure non-empty partitions
        if train_end == 0 or val_end <= train_end or val_end >= n:
            raise ValueError(
                f"Split ratios resulted in an empty slice with n={n}. Adjust ratios or provide more data."
            )

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        # Strict chronological assertions: no temporal leakage or overlap
        train_max = pd.to_datetime(train_df.index.max())
        val_min = pd.to_datetime(val_df.index.min())
        val_max = pd.to_datetime(val_df.index.max())
        test_min = pd.to_datetime(test_df.index.min())

        if not (train_max < val_min):
            raise AssertionError(
                f"Time-series leak detected! Train max ({train_max}) is not strictly before Val min ({val_min})."
            )

        if not (val_max < test_min):
            raise AssertionError(
                f"Time-series leak detected! Val max ({val_max}) is not strictly before Test min ({test_min})."
            )

        result = TimeSeriesSplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            train_range=(pd.to_datetime(train_df.index.min()), train_max),
            val_range=(val_min, val_max),
            test_range=(test_min, pd.to_datetime(test_df.index.max())),
        )

        logger.info(
            f"Chronological split completed:\n"
            f"  Train: {len(train_df)} rows [{result.train_range[0].strftime('%Y-%m-%d')} -> {train_max.strftime('%Y-%m-%d')}]\n"
            f"  Val:   {len(val_df)} rows [{val_min.strftime('%Y-%m-%d')} -> {val_max.strftime('%Y-%m-%d')}]\n"
            f"  Test:  {len(test_df)} rows [{test_min.strftime('%Y-%m-%d')} -> {result.test_range[1].strftime('%Y-%m-%d')}]"
        )

        return result

    @staticmethod
    def train_test_by_date(
        df: pd.DataFrame,
        split_date: str,
        val_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """Splits a dataset chronologically using explicit cutoff dates.

        Args:
            df: DataFrame with DatetimeIndex.
            split_date: Timestamp string (YYYY-MM-DD) designating start of test set.
            val_date: Optional timestamp string designating start of validation set.

        Returns:
            Tuple of (train_df, val_df, test_df) or (train_df, None, test_df) if val_date is None.
        """
        df_sorted = df.sort_index(ascending=True)
        split_dt = pd.to_datetime(split_date)

        if val_date:
            val_dt = pd.to_datetime(val_date)
            if not (val_dt < split_dt):
                raise ValueError(f"val_date ({val_date}) must be earlier than split_date ({split_date}).")
            train_df = df_sorted[df_sorted.index < val_dt].copy()
            val_df = df_sorted[(df_sorted.index >= val_dt) & (df_sorted.index < split_dt)].copy()
            test_df = df_sorted[df_sorted.index >= split_dt].copy()
            return train_df, val_df, test_df

        train_df = df_sorted[df_sorted.index < split_dt].copy()
        test_df = df_sorted[df_sorted.index >= split_dt].copy()
        return train_df, None, test_df

    @staticmethod
    def walk_forward_splits(
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 60,
        min_train_size: Optional[int] = None,
        expanding: bool = True,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Generates walk-forward (expanding or rolling) train/test folds for time-series cross-validation.

        Args:
            df: DataFrame sorted chronologically.
            n_splits: Number of walk-forward folds.
            test_size: Number of time steps per test fold.
            min_train_size: Minimum number of rows in first training window.
            expanding: If True, training window expands each fold; if False, window rolls (fixed size).

        Yields:
            Tuple of (train_fold, test_fold).
        """
        df_sorted = df.sort_index(ascending=True)
        n = len(df_sorted)
        total_test = n_splits * test_size

        if total_test >= n:
            raise ValueError(
                f"n_splits ({n_splits}) * test_size ({test_size}) = {total_test} exceeds total samples {n}."
            )

        initial_train_size = min_train_size or (n - total_test)
        if initial_train_size < 10:
            raise ValueError(f"Initial train size ({initial_train_size}) is too small.")

        for i in range(n_splits):
            test_start = initial_train_size + i * test_size
            test_end = test_start + test_size

            if expanding:
                train_fold = df_sorted.iloc[:test_start].copy()
            else:
                train_fold = df_sorted.iloc[i * test_size:test_start].copy()

            test_fold = df_sorted.iloc[test_start:test_end].copy()

            # Ensure chronological order
            assert train_fold.index.max() < test_fold.index.min(), "Leakage in walk-forward fold"
            yield train_fold, test_fold
