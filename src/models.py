"""Baseline Machine Learning models for Northgate AI Stock Predictor.

Trains and evaluates Linear Regression, Random Forest, and XGBoost models
to forecast next-day closing stock prices using chronological 80/20 time-based splits.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.data_loader import DEFAULT_TICKERS, load_raw_data
from src.features import engineer_features

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("models")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_CSV_PATH = PROCESSED_DIR / "baseline_results.csv"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes RMSE, MAE, and MAPE between true and predicted prices.

    Args:
        y_true: Ground truth target prices.
        y_pred: Predicted target prices.

    Returns:
        Dictionary with 'RMSE', 'MAE', and 'MAPE_%'.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_%": round(mape, 4),
    }


def split_time_series_80_20(
    df: pd.DataFrame,
    target_col: str = "Target_Next_Close",
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Partitions dataset into chronological 80% train and 20% test sets.

    Conforms to Rule 7: Time-series data must be split chronologically, never randomly.

    Args:
        df: DataFrame containing features and target column.
        target_col: Name of the target variable to forecast.
        train_ratio: Proportion of earliest historical data for training (default 0.80).

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_columns).

    Raises:
        ValueError: If target_col missing or data length is inadequate.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Sort strictly ascending by time
    df_sorted = df.sort_index(ascending=True)

    # Exclude non-predictive metadata and target column from features
    exclude_cols = {target_col, "Dividends", "Stock Splits"}
    feature_cols = [c for c in df_sorted.columns if c not in exclude_cols]

    n_samples = len(df_sorted)
    split_idx = int(n_samples * train_ratio)

    train_data = df_sorted.iloc[:split_idx]
    test_data = df_sorted.iloc[split_idx:]

    # Rule 7 Assertion: Verify zero lookahead leakage
    assert train_data.index.max() < test_data.index.min(), (
        f"Temporal leakage detected! Train max ({train_data.index.max()}) "
        f"is not strictly before Test min ({test_data.index.min()})"
    )

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    logger.info(
        f"Chronological 80/20 Split: Train={len(X_train)} samples "
        f"[{train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}], "
        f"Test={len(X_test)} samples "
        f"[{test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}]."
    )

    return X_train, y_train, X_test, y_test, feature_cols


def get_baseline_models() -> Dict[str, Any]:
    """Instantiates the candidate baseline models for comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_ticker(
    ticker: str,
    feature_df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Trains all baseline models for a specific stock ticker and evaluates performance.

    Args:
        ticker: Stock symbol (e.g. 'RELIANCE.NS').
        feature_df: DataFrame output from engineer_features().
        train_ratio: Train ratio (0.80).

    Returns:
        Tuple of (results_list, trained_models_dict).
    """
    logger.info(f"=== Training Baseline Models for {ticker} ===")
    X_train, y_train, X_test, y_test, feature_cols = split_time_series_80_20(
        feature_df,
        target_col="Target_Next_Close",
        train_ratio=train_ratio,
    )

    models = get_baseline_models()
    ticker_results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for model_name, model in models.items():
        logger.info(f"Fitting {model_name} on {ticker}...")
        model.fit(X_train, y_train)

        # In-sample (train) metrics
        train_preds = model.predict(X_train)
        train_metrics = compute_metrics(y_train.values, train_preds)

        # Out-of-sample (test) metrics
        test_preds = model.predict(X_test)
        test_metrics = compute_metrics(y_test.values, test_preds)

        logger.info(
            f"[{ticker}] {model_name:<18} -> Test RMSE: {test_metrics['RMSE']:>7.2f} | "
            f"Test MAE: {test_metrics['MAE']:>7.2f} | Test MAPE: {test_metrics['MAPE_%']:>5.2f}%"
        )

        ticker_results.append(
            {
                "Ticker": ticker,
                "Model": model_name,
                "Train_RMSE": train_metrics["RMSE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_MAE": test_metrics["MAE"],
                "Test_MAPE_%": test_metrics["MAPE_%"],
            }
        )

        trained_models[model_name] = {
            "model": model,
            "metrics": test_metrics,
            "feature_cols": feature_cols,
        }

    return ticker_results, trained_models


def run_baseline_experiments(
    tickers: Optional[List[str]] = None,
    save_results: bool = True,
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """Runs end-to-end baseline modeling experiments across target tickers.

    Args:
        tickers: List of tickers to evaluate (defaults to DEFAULT_TICKERS).
        save_results: Whether to save comparison table to data/processed/baseline_results.csv.

    Returns:
        Tuple of (results_df, best_model_instance, best_model_meta).
    """
    tickers_to_run = tickers or DEFAULT_TICKERS
    all_results: List[Dict[str, Any]] = []
    all_trained_models: Dict[str, Dict[str, Any]] = {}

    for ticker in tickers_to_run:
        try:
            raw_df = load_raw_data(ticker)
            feature_df = engineer_features(raw_df, include_target=True)
            results, trained_models = train_and_evaluate_ticker(ticker, feature_df)
            all_results.extend(results)
            all_trained_models[ticker] = trained_models
        except Exception as e:
            logger.error(f"Error processing baseline models for {ticker}: {e}")

    results_df = pd.DataFrame(all_results)

    if save_results and not results_df.empty:
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        logger.info(f"Saved baseline comparison table to {RESULTS_CSV_PATH}")

    # Determine overall best model based on lowest mean Test MAPE across tickers
    model_performance = results_df.groupby("Model")["Test_MAPE_%"].mean().reset_index()
    best_model_name = model_performance.sort_values(by="Test_MAPE_%").iloc[0]["Model"]
    best_mape = model_performance.sort_values(by="Test_MAPE_%").iloc[0]["Test_MAPE_%"]

    logger.info(
        f"Top overall baseline model across all tickers: {best_model_name} (Mean MAPE: {best_mape:.2f}%)"
    )

    # Select primary stock model (RELIANCE.NS or first available) as representative saved artifact
    primary_ticker = tickers_to_run[0]
    best_model_entry = all_trained_models[primary_ticker][best_model_name]
    best_model_obj = best_model_entry["model"]

    best_model_meta = {
        "best_model_name": best_model_name,
        "primary_ticker": primary_ticker,
        "mean_mape_pct": float(best_mape),
        "primary_ticker_metrics": best_model_entry["metrics"],
        "feature_count": len(best_model_entry["feature_cols"]),
        "features": best_model_entry["feature_cols"],
    }

    # Save best model artifact to models/
    model_save_path = MODELS_DIR / "best_model.joblib"
    meta_save_path = MODELS_DIR / "best_model_metadata.json"

    joblib.dump(best_model_obj, model_save_path)
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(best_model_meta, f, indent=4)

    logger.info(f"Saved best model ({best_model_name}) to {model_save_path}")
    logger.info(f"Saved model metadata to {meta_save_path}")

    return results_df, best_model_obj, best_model_meta


if __name__ == "__main__":
    results_table, _, meta = run_baseline_experiments()
    print("\n" + "=" * 75)
    print("NORTHGATE AI STOCK PREDICTOR - BASELINE ML RESULTS (80/20 Time Split)")
    print("=" * 75)
    print(results_table.to_string(index=False))
    print("=" * 75)
    print(
        f"Best Model Selected: {meta['best_model_name']} (Saved to models/best_model.joblib)"
    )
    print("=" * 75)
