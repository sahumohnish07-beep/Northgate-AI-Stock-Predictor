"""Feature engineering module for Northgate AI Stock Predictor.

Computes technical indicators:
- RSI (Relative Strength Index)
- MACD (Line, Signal, Histogram)
- Moving Averages: SMA and EMA (10, 20, 50)
- Bollinger Bands (Upper, Middle, Lower, Bandwidth)
- Returns & Historical Rolling Volatility
- Lagged Price, Return, and Volume features
- Next-day Close target generation for supervised forecasting
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI) using exponential smoothing.

    Args:
        series: Price series (typically Close price).
        period: Lookback window period (default 14).

    Returns:
        pd.Series of RSI values ranging from 0 to 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (exponential moving average with alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculates Moving Average Convergence Divergence (MACD).

    Args:
        series: Price series (Close).
        fast_period: Fast EMA span (default 12).
        slow_period: Slow EMA span (default 26).
        signal_period: Signal line EMA span (default 9).

    Returns:
        pd.DataFrame with ['MACD', 'MACD_Signal', 'MACD_Hist'].
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram,
        },
        index=series.index,
    )


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Calculates Bollinger Bands and Bandwidth.

    Args:
        series: Price series (Close).
        period: Window size (default 20).
        num_std: Multiplier for standard deviation (default 2.0).

    Returns:
        pd.DataFrame with ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'].
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    width = (upper - lower) / middle.replace(0, np.nan)

    return pd.DataFrame(
        {
            "BB_Upper": upper,
            "BB_Middle": middle,
            "BB_Lower": lower,
            "BB_Width": width,
        },
        index=series.index,
    )


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculates Simple Moving Average (SMA) for a given lookback window.

    Args:
        series: Price series (typically Close price).
        period: Window length (default 20).

    Returns:
        pd.Series of SMA values.
    """
    return series.rolling(window=period).mean()


def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculates Exponential Moving Average (EMA) for a given lookback span.

    Args:
        series: Price series (typically Close price).
        period: Exponential span (default 20).

    Returns:
        pd.Series of EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_moving_averages(
    series: pd.Series,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calculates Simple (SMA) and Exponential (EMA) Moving Averages.

    Args:
        series: Price series (Close).
        windows: List of window lengths (defaults to [10, 20, 50]).

    Returns:
        pd.DataFrame containing SMA_<w> and EMA_<w> columns.
    """
    if windows is None:
        windows = [10, 20, 50]

    ma_dict = {}
    for w in windows:
        ma_dict[f"SMA_{w}"] = compute_sma(series, period=w)
        ma_dict[f"EMA_{w}"] = compute_ema(series, period=w)

    return pd.DataFrame(ma_dict, index=series.index)


def compute_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generates time-lagged features for specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to lag (defaults to ['Close', 'Daily_Return', 'Volume']).
        lags: List of lag intervals (defaults to [1, 2, 3, 5]).

    Returns:
        pd.DataFrame containing lagged columns.
    """
    if columns is None:
        columns = ["Close", "Daily_Return", "Volume"]
    if lags is None:
        lags = [1, 2, 3, 5]

    lag_dict = {}
    for col in columns:
        if col in df.columns:
            for lag in lags:
                lag_dict[f"{col}_Lag_{lag}"] = df[col].shift(lag)

    return pd.DataFrame(lag_dict, index=df.index)


def engineer_features(
    df: pd.DataFrame,
    include_target: bool = True,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """Builds a comprehensive technical feature matrix from OHLCV data.

    Calculates:
    - Daily Returns & Rolling Volatility (10d, 20d annualized)
    - RSI (14)
    - MACD, Signal, Histogram
    - SMA & EMA for 10, 20, 50 periods
    - Bollinger Bands (Upper, Middle, Lower, Width)
    - Lag features (Close, Return, Volume lags 1, 2, 3, 5)
    - Target: Next-day Close (Target_Next_Close)

    Args:
        df: Input OHLCV DataFrame with standard columns ('Open', 'High', 'Low', 'Close', 'Volume').
        include_target: Whether to attach the future Target_Next_Close column.
        target_horizon: Number of days forward to predict (default 1 = next-day close).

    Returns:
        pd.DataFrame with all features, cleaned of NaN values, sorted chronologically.
    """
    data = df.copy()
    data = data.sort_index(ascending=True)

    # 1. Daily Returns & Rolling Volatility
    data["Daily_Return"] = data["Close"].pct_change().fillna(0.0)
    data["Volatility_10"] = data["Daily_Return"].rolling(window=10).std() * np.sqrt(252)
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

    # 2. RSI (14)
    data["RSI_14"] = compute_rsi(data["Close"], period=14)

    # 3. MACD
    macd_df = compute_macd(data["Close"])
    for col in macd_df.columns:
        data[col] = macd_df[col]

    # 4. Moving Averages (SMA/EMA 10, 20, 50)
    ma_df = compute_moving_averages(data["Close"], windows=[10, 20, 50])
    for col in ma_df.columns:
        data[col] = ma_df[col]

    # 5. Bollinger Bands
    bb_df = compute_bollinger_bands(data["Close"], period=20)
    for col in bb_df.columns:
        data[col] = bb_df[col]

    # 6. Lag Features
    lag_df = compute_lag_features(data, columns=["Close", "Daily_Return", "Volume"], lags=[1, 2, 3, 5])
    for col in lag_df.columns:
        data[col] = lag_df[col]

    # 7. Supervised Target: Next-Day Close
    if include_target:
        data["Target_Next_Close"] = data["Close"].shift(-target_horizon)

    # Clean NaNs caused by moving windows and lags
    cleaned_df = data.dropna().copy()
    return cleaned_df

