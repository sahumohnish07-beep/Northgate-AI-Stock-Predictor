"""
Portfolio Optimization Module for Northgate AI Stock Predictor.

Applies Modern Portfolio Theory (MPT) using PyPortfolioOpt to calculate expected returns,
risk covariance matrices, and optimal portfolio weight allocations across different investor
risk profiles (Low Risk: Minimum Volatility, Medium Risk: Maximum Sharpe Ratio,
and High Risk: Maximum Return with single-asset weight cap).
"""

from collections import OrderedDict
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
import pypfopt.plotting as pplot

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
DEFAULT_RISK_FREE_RATE = 0.06  # 6.0% benchmark for Indian sovereign 10Y G-Sec
DEFAULT_WEIGHT_CAP = 0.40       # 40% maximum allocation for high-risk profile


def load_historical_prices(
    tickers: Optional[List[str]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Load historical closing prices for a list of tickers from data/raw/ CSV files.

    Parameters
    ----------
    tickers : Optional[List[str]]
        List of stock ticker symbols to load. Defaults to DEFAULT_TICKERS.
    raw_dir : Optional[Union[str, Path]]
        Directory where CSV files are saved. Defaults to data/raw.
    price_col : str
        Price column to extract ('Adj Close' or 'Close'). Defaults to 'Adj Close'.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with ticker columns containing historical prices.

    Raises
    ------
    FileNotFoundError
        If data/raw CSV file does not exist for any requested ticker.
    ValueError
        If price_col is missing or merged data contains fewer than 30 observations.
    """
    selected_tickers = tickers or DEFAULT_TICKERS
    data_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR

    series_dict: Dict[str, pd.Series] = {}

    for ticker in selected_tickers:
        file_path = data_dir / f"{ticker}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Historical data file not found: {file_path}. "
                f"Please run src/data_loader.py to collect market data."
            )

        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"])
        df = df.set_index("Date")

        col_to_use = price_col if price_col in df.columns else "Close"
        if col_to_use not in df.columns:
            raise ValueError(f"Neither '{price_col}' nor 'Close' found in {file_path}")

        series = df[col_to_use].astype(float)
        series_dict[ticker] = series

    prices_df = pd.DataFrame(series_dict).dropna()

    if len(prices_df) < 30:
        raise ValueError(
            f"Insufficient aligned trading days ({len(prices_df)}). Minimum 30 required."
        )

    return prices_df


def compute_expected_returns_and_covariance(
    prices_df: pd.DataFrame,
    return_method: str = "mean",
    cov_method: str = "sample",
    frequency: int = 252,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate annualized expected returns vector and covariance matrix from historical prices.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Historical price series with DateTime index and ticker columns.
    return_method : str
        Method for expected returns: 'mean' (historical mean) or 'ema' (exponential).
    cov_method : str
        Method for covariance matrix: 'sample' or 'ledoit_wolf' (shrinkage).
    frequency : int
        Number of trading sessions per year. Defaults to 252.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Tuple of (expected_returns_vector, covariance_matrix).
    """
    if prices_df.empty or len(prices_df.columns) < 2:
        raise ValueError("prices_df must contain at least 2 tickers and non-empty rows.")

    # Calculate expected returns
    if return_method == "ema":
        mu = expected_returns.ema_historical_return(prices_df, frequency=frequency)
    else:
        mu = expected_returns.mean_historical_return(prices_df, frequency=frequency)

    # Calculate covariance matrix
    if cov_method == "ledoit_wolf":
        cov_matrix = risk_models.CovarianceShrinkage(
            prices_df, frequency=frequency
        ).ledoit_wolf()
    else:
        cov_matrix = risk_models.sample_cov(prices_df, frequency=frequency)

    return mu, cov_matrix


def optimize_portfolio(
    prices_df: Optional[pd.DataFrame] = None,
    mu: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    risk_profile: str = "Medium",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
) -> Dict[str, Any]:
    """
    Optimize portfolio weights based on the specified investor risk profile.

    Profiles:
      - 'Low' / 'min_volatility': Minimum volatility portfolio (conservative).
      - 'Medium' / 'max_sharpe': Maximum Sharpe ratio portfolio (tangency portfolio).
      - 'High' / 'max_return': Maximum return subject to a single-asset weight cap.

    Parameters
    ----------
    prices_df : Optional[pd.DataFrame]
        Historical prices (used if mu and cov_matrix are not provided).
    mu : Optional[pd.Series]
        Expected annual returns vector.
    cov_matrix : Optional[pd.DataFrame]
        Annualized covariance matrix.
    risk_profile : str
        Risk profile: 'Low', 'Medium', or 'High'. Defaults to 'Medium'.
    risk_free_rate : float
        Risk-free rate benchmark. Defaults to 0.06 (6.0%).
    weight_cap : float
        Maximum allocation allowed for any single asset under the High risk profile.
        Defaults to 0.40 (40%).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing profile name, cleaned weights, expected return,
        annual volatility, Sharpe ratio, and input parameters.
    """
    # Compute mu and cov_matrix if not explicitly supplied
    if mu is None or cov_matrix is None:
        if prices_df is None:
            prices_df = load_historical_prices()
        mu, cov_matrix = compute_expected_returns_and_covariance(prices_df)

    num_assets = len(mu)
    norm_profile = risk_profile.strip().lower()

    if norm_profile in ["low", "min_volatility", "conservative"]:
        profile_name = "Low Risk (Min Volatility)"
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
        ef.min_volatility()

    elif norm_profile in ["medium", "max_sharpe", "balanced"]:
        profile_name = "Medium Risk (Max Sharpe)"
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception:
            # Fallback to quadratic utility or min_volatility if all expected returns < rf
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
            ef.min_volatility()

    elif norm_profile in ["high", "max_return", "aggressive"]:
        profile_name = f"High Risk (Max Return, Cap {weight_cap:.0%})"
        if weight_cap < (1.0 / num_assets):
            raise ValueError(
                f"weight_cap ({weight_cap:.2f}) must be >= 1 / num_assets "
                f"({1.0 / num_assets:.2f}) to ensure a feasible allocation."
            )
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, weight_cap))
        try:
            ef._max_return(return_value=False)
        except Exception:
            # In case _max_return throws, solve using max_sharpe with capped weights
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, weight_cap))
            ef.max_sharpe(risk_free_rate=risk_free_rate)
    else:
        raise ValueError(
            f"Unknown risk profile '{risk_profile}'. Must be 'Low', 'Medium', or 'High'."
        )

    raw_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
    exp_return, volatility, sharpe = ef.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    return {
        "risk_profile": profile_name,
        "raw_profile_key": risk_profile,
        "weights": dict(raw_weights),
        "expected_return": float(exp_return),
        "annual_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "risk_free_rate": risk_free_rate,
        "weight_cap": weight_cap if "high" in norm_profile else 1.0,
    }


def optimize_all_profiles(
    prices_df: Optional[pd.DataFrame] = None,
    mu: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
) -> Dict[str, Dict[str, Any]]:
    """
    Run optimization across all three standard risk profiles: Low, Medium, and High.

    Parameters
    ----------
    prices_df : Optional[pd.DataFrame]
        Historical prices DataFrame.
    mu : Optional[pd.Series]
        Expected returns vector.
    cov_matrix : Optional[pd.DataFrame]
        Covariance matrix.
    risk_free_rate : float
        Annual risk-free rate benchmark.
    weight_cap : float
        Single-asset allocation cap for High risk profile.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with keys 'Low', 'Medium', 'High' containing optimization details.
    """
    if mu is None or cov_matrix is None:
        if prices_df is None:
            prices_df = load_historical_prices()
        mu, cov_matrix = compute_expected_returns_and_covariance(prices_df)

    results: Dict[str, Dict[str, Any]] = {}
    for profile in ["Low", "Medium", "High"]:
        results[profile] = optimize_portfolio(
            mu=mu,
            cov_matrix=cov_matrix,
            risk_profile=profile,
            risk_free_rate=risk_free_rate,
            weight_cap=weight_cap,
        )

    return results


def export_portfolio_allocations(
    profile_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Construct a consolidated allocation and performance table and save to CSV.

    Parameters
    ----------
    profile_results : Dict[str, Dict[str, Any]]
        Optimization output dictionary from optimize_all_profiles.
    output_path : Optional[Union[str, Path]]
        Destination path for CSV export. Defaults to data/processed/portfolio_allocations.csv.

    Returns
    -------
    pd.DataFrame
        Formatted summary DataFrame.
    """
    dest_path = (
        Path(output_path)
        if output_path
        else PROCESSED_DATA_DIR / "portfolio_allocations.csv"
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for key, data in profile_results.items():
        row = {
            "Risk_Profile": data["risk_profile"],
            "Profile_Key": key,
            "Expected_Return": data["expected_return"],
            "Annual_Volatility": data["annual_volatility"],
            "Sharpe_Ratio": data["sharpe_ratio"],
        }
        for ticker, weight in data["weights"].items():
            row[f"Weight_{ticker}"] = weight
        records.append(row)

    df_allocations = pd.DataFrame(records)
    df_allocations.to_csv(dest_path, index=False)
    return df_allocations


def generate_recommendations(
    profile_results: Dict[str, Dict[str, Any]],
    equal_weight_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Generate algorithmic Buy / Hold / Sell / Underweight recommendations for each stock
    based on average weight allocation across risk profiles relative to an equal-weight baseline.

    Parameters
    ----------
    profile_results : Dict[str, Dict[str, Any]]
        Multi-profile optimization results.
    equal_weight_threshold : float
        Benchmark equal allocation (e.g., 0.20 for 5 stocks).

    Returns
    -------
    pd.DataFrame
        DataFrame with Ticker, Mean_Weight, Low_Weight, Med_Weight, High_Weight, and Recommendation.
    """
    tickers = list(next(iter(profile_results.values()))["weights"].keys())
    rows = []

    for ticker in tickers:
        w_low = profile_results.get("Low", {}).get("weights", {}).get(ticker, 0.0)
        w_med = profile_results.get("Medium", {}).get("weights", {}).get(ticker, 0.0)
        w_high = profile_results.get("High", {}).get("weights", {}).get(ticker, 0.0)
        mean_w = (w_low + w_med + w_high) / 3.0

        if mean_w >= equal_weight_threshold * 1.25:
            recommendation = "Strong Buy (Overweight)"
        elif mean_w >= equal_weight_threshold * 0.75:
            recommendation = "Buy / Moderate Allocation"
        elif mean_w > 0.02:
            recommendation = "Hold / Selective"
        else:
            recommendation = "Underweight / Avoid"

        rows.append(
            {
                "Ticker": ticker,
                "Mean_Weight": round(mean_w, 4),
                "Low_Risk_Weight": round(w_low, 4),
                "Medium_Risk_Weight": round(w_med, 4),
                "High_Risk_Weight": round(w_high, 4),
                "Recommendation": recommendation,
            }
        )

    rec_df = pd.DataFrame(rows).sort_values("Mean_Weight", ascending=False)
    return rec_df


def plot_efficient_frontier_and_profiles(
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    profile_results: Optional[Dict[str, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Path:
    """
    Generate an Efficient Frontier chart overlaid with individual asset risk/returns
    and markers for the Low, Medium, and High risk optimal portfolios.

    Parameters
    ----------
    mu : pd.Series
        Expected returns vector.
    cov_matrix : pd.DataFrame
        Covariance matrix.
    profile_results : Optional[Dict[str, Dict[str, Any]]]
        Dictionary with profile optimization details. Computed if None.
    output_path : Optional[Union[str, Path]]
        File path to save the output figure. Defaults to data/processed/efficient_frontier.png.
    risk_free_rate : float
        Benchmark risk-free rate.

    Returns
    -------
    Path
        Absolute path to the saved figure image.
    """
    dest_path = (
        Path(output_path)
        if output_path
        else PROCESSED_DATA_DIR / "efficient_frontier.png"
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if profile_results is None:
        profile_results = optimize_all_profiles(
            mu=mu, cov_matrix=cov_matrix, risk_free_rate=risk_free_rate
        )

    fig, ax = plt.subplots(figsize=(11, 7))

    # Base Efficient Frontier curve from PyPortfolioOpt
    ef_curve = EfficientFrontier(mu, cov_matrix)
    pplot.plot_efficient_frontier(
        ef_curve,
        ax=ax,
        show_assets=True,
        show_tickers=True,
        points=100,
    )

    # Plot Low Risk profile
    low = profile_results["Low"]
    ax.scatter(
        low["annual_volatility"],
        low["expected_return"],
        marker="s",
        s=160,
        color="#1f77b4",
        edgecolors="black",
        linewidth=1.5,
        zorder=6,
        label=(
            f"Low Risk (Min Vol): Ret {low['expected_return']:.1%}, "
            f"Vol {low['annual_volatility']:.1%}, Sharpe {low['sharpe_ratio']:.2f}"
        ),
    )

    # Plot Medium Risk profile
    med = profile_results["Medium"]
    ax.scatter(
        med["annual_volatility"],
        med["expected_return"],
        marker="*",
        s=260,
        color="#2ca02c",
        edgecolors="black",
        linewidth=1.5,
        zorder=7,
        label=(
            f"Medium Risk (Max Sharpe): Ret {med['expected_return']:.1%}, "
            f"Vol {med['annual_volatility']:.1%}, Sharpe {med['sharpe_ratio']:.2f}"
        ),
    )

    # Plot High Risk profile
    high = profile_results["High"]
    ax.scatter(
        high["annual_volatility"],
        high["expected_return"],
        marker="^",
        s=180,
        color="#d62728",
        edgecolors="black",
        linewidth=1.5,
        zorder=6,
        label=(
            f"High Risk (Max Ret, Cap 40%): Ret {high['expected_return']:.1%}, "
            f"Vol {high['annual_volatility']:.1%}, Sharpe {high['sharpe_ratio']:.2f}"
        ),
    )

    # Risk-free rate reference line
    ax.axhline(
        y=risk_free_rate,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Risk-Free Benchmark ({risk_free_rate:.1%})",
    )

    ax.set_title(
        "Efficient Frontier & Optimal Asset Allocations (Northgate AI)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Expected Annual Return", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(dest_path, dpi=300)
    plt.close(fig)

    return dest_path


def main() -> None:
    """CLI execution entrypoint for portfolio optimization."""
    print("=" * 70)
    print("Northgate AI Stock Predictor - Phase 5: Portfolio Optimizer")
    print("=" * 70)

    print("\n[1/5] Loading historical prices for default stock universe...")
    prices = load_historical_prices(DEFAULT_TICKERS)
    print(
        f"  Loaded {len(prices)} trading days for {len(prices.columns)} assets: "
        f"{', '.join(prices.columns)}"
    )

    print("\n[2/5] Computing expected returns and covariance matrix...")
    mu, cov_matrix = compute_expected_returns_and_covariance(prices)
    for ticker in mu.index:
        vol = np.sqrt(cov_matrix.loc[ticker, ticker])
        print(f"  {ticker:12s} | Exp Return: {mu[ticker]:+7.2%} | Annual Vol: {vol:6.2%}")

    print("\n[3/5] Optimizing portfolios across risk profiles...")
    profile_results = optimize_all_profiles(
        mu=mu,
        cov_matrix=cov_matrix,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        weight_cap=DEFAULT_WEIGHT_CAP,
    )

    for key, data in profile_results.items():
        print(f"\n--- {data['risk_profile']} ---")
        print(
            f"  Expected Return:   {data['expected_return']:.2%}\n"
            f"  Annual Volatility: {data['annual_volatility']:.2%}\n"
            f"  Sharpe Ratio:      {data['sharpe_ratio']:.2f}"
        )
        print("  Allocations:")
        for t, w in data["weights"].items():
            print(f"    {t:12s}: {w:6.2%}")

    print("\n[4/5] Exporting allocation metrics and recommendations...")
    alloc_df = export_portfolio_allocations(profile_results)
    rec_df = generate_recommendations(profile_results)
    print(f"  Saved allocations to: {PROCESSED_DATA_DIR / 'portfolio_allocations.csv'}")
    print("\nAlgorithmic Stock Allocations & Recommendations:")
    print(rec_df.to_string(index=False))

    print("\n[5/5] Generating Efficient Frontier visualization...")
    plot_path = plot_efficient_frontier_and_profiles(
        mu=mu,
        cov_matrix=cov_matrix,
        profile_results=profile_results,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
    )
    print(f"  Saved plot to: {plot_path}")

    print("\n" + "=" * 70)
    print("Portfolio optimization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
