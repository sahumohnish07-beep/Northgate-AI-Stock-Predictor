"""
Algorithmic Stock Recommendation Engine and Backtesting Module for Northgate AI.

Combines 5-day predictive forward return forecasts, FinBERT financial news sentiment,
and market volatility into an actionable Buy / Hold / Sell decision engine with
risk-profile threshold adaptability (Low, Medium, High). Includes a walk-forward,
lookahead-free backtesting framework comparing active strategy returns against
passive Buy-and-Hold benchmarks.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_raw_data

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_RISK_FREE_RATE = 0.06  # 6.0% benchmark for Indian 10Y sovereign G-Sec

# Risk Profile Threshold Configurations
# Base rule (Medium): Buy if ret > +2% and sent > 0.1; Sell if ret < -2% and sent < -0.1; else Hold
RISK_PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Low": {
        "buy_return": 0.03,  # +3.0%: Requires stronger positive return hurdle
        "buy_sentiment": 0.15,  # +0.15: Requires distinctly positive sentiment
        "sell_return": -0.015,  # -1.5%: Tighter stop to cut losses rapidly
        "sell_sentiment": -0.05,  # -0.05: Quicker exit on mild negative news
        "volatility_tolerance": 0.22,  # 22% annualized volatility cap
    },
    "Medium": {
        "buy_return": 0.02,  # +2.0%: Standard benchmark entry hurdle
        "buy_sentiment": 0.10,  # +0.10: Positive news sentiment
        "sell_return": -0.02,  # -2.0%: Standard stop hurdle
        "sell_sentiment": -0.10,  # -0.10: Negative news sentiment
        "volatility_tolerance": 0.30,  # 30% annualized volatility cap
    },
    "High": {
        "buy_return": 0.01,  # +1.0%: Aggressive entry hurdle
        "buy_sentiment": 0.00,  #  0.00: Neutral or mildly positive news
        "sell_return": -0.03,  # -3.0%: Wider drawdown tolerance
        "sell_sentiment": -0.15,  # -0.15: Deeper negative news required to exit
        "volatility_tolerance": 0.45,  # 45% annualized volatility cap
    },
}


def get_risk_thresholds(risk_profile: str = "Medium") -> Dict[str, float]:
    """
    Retrieve signal thresholds tailored to the investor risk profile.

    Parameters
    ----------
    risk_profile : str
        Risk profile name: 'Low' (conservative), 'Medium' (balanced), or 'High' (aggressive).

    Returns
    -------
    Dict[str, float]
        Dictionary with buy_return, buy_sentiment, sell_return, sell_sentiment,
        and volatility_tolerance.

    Raises
    ------
    ValueError
        If an unrecognised risk profile string is provided.
    """
    norm_profile = risk_profile.strip().capitalize()
    alias_map = {
        "Conservative": "Low",
        "Balanced": "Medium",
        "Aggressive": "High",
    }
    canonical_profile = alias_map.get(norm_profile, norm_profile)

    if canonical_profile not in RISK_PROFILE_THRESHOLDS:
        raise ValueError(
            f"Unknown risk profile '{risk_profile}'. Must be 'Low', 'Medium', or 'High'."
        )

    return RISK_PROFILE_THRESHOLDS[canonical_profile].copy()


def generate_recommendation(
    predicted_return_5d: float,
    sentiment_score: float,
    volatility: float,
    risk_profile: str = "Medium",
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate multi-factor inputs to generate a Buy / Hold / Sell recommendation,
    an estimated confidence level (0.0 to 1.0), and an explanatory rationale string.

    Scoring Logic:
      - Buy: predicted_return_5d > buy_return AND sentiment_score > buy_sentiment
      - Sell: predicted_return_5d < sell_return AND sentiment_score < sell_sentiment
      - Otherwise Hold
      - Under Low risk, excessive volatility can temper a Buy into a Hold for capital preservation.

    Parameters
    ----------
    predicted_return_5d : float
        Forecasted return over the next 5 trading days (e.g., 0.035 for +3.5%).
    sentiment_score : float
        Aggregated FinBERT news sentiment polarity score in [-1.0, +1.0].
    volatility : float
        Historical or annualized volatility (e.g., 0.22 for 22%).
    risk_profile : str
        Investor risk appetite: 'Low', 'Medium', or 'High'. Defaults to 'Medium'.
    custom_thresholds : Optional[Dict[str, float]]
        Optional manual override dictionary for signal thresholds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - recommendation: 'Buy', 'Hold', or 'Sell'
          - confidence: float between 0.0 and 1.0
          - reason: explanatory string with numerical reasoning
          - risk_profile: normalized risk profile
          - metrics: dictionary of input metrics
          - thresholds: dictionary of active thresholds applied
    """
    thresholds = custom_thresholds or get_risk_thresholds(risk_profile)
    b_ret = thresholds["buy_return"]
    b_sent = thresholds["buy_sentiment"]
    s_ret = thresholds["sell_return"]
    s_sent = thresholds["sell_sentiment"]
    vol_tol = thresholds["volatility_tolerance"]

    canonical_profile = risk_profile.strip().capitalize()
    if canonical_profile in ["Conservative", "Low"]:
        canonical_profile = "Low"
    elif canonical_profile in ["Balanced", "Medium"]:
        canonical_profile = "Medium"
    elif canonical_profile in ["Aggressive", "High"]:
        canonical_profile = "High"

    # Evaluate base rules
    is_buy_candidate = (predicted_return_5d > b_ret) and (sentiment_score > b_sent)
    is_sell_candidate = (predicted_return_5d < s_ret) and (sentiment_score < s_sent)

    # Low risk constraint: excessive volatility prevents aggressive entry
    is_volatility_exceeded = (canonical_profile == "Low") and (
        volatility > vol_tol * 1.25
    )

    if is_buy_candidate and not is_volatility_exceeded:
        recommendation = "Buy"

        # Confidence calculation for Buy
        ret_excess = max(0.0, predicted_return_5d - b_ret)
        sent_excess = max(0.0, sentiment_score - b_sent)
        ret_boost = min(0.18, (ret_excess / 0.04) * 0.18)
        sent_boost = min(0.12, (sent_excess / 0.40) * 0.12)
        vol_penalty = (
            max(0.0, (volatility - vol_tol) / 0.20) * 0.10
            if volatility > vol_tol
            else 0.0
        )

        confidence = float(
            np.clip(0.68 + ret_boost + sent_boost - vol_penalty, 0.50, 0.98)
        )
        reason = (
            f"Bullish conviction ({canonical_profile} Risk): Predicted 5-day return "
            f"({predicted_return_5d:+.2%}) exceeds the {b_ret:+.2%} threshold with positive news "
            f"sentiment ({sentiment_score:+.2f} > {b_sent:+.2f}) and controlled volatility ({volatility:.1%})."
        )

    elif is_sell_candidate:
        recommendation = "Sell"

        # Confidence calculation for Sell
        ret_deficit = max(0.0, s_ret - predicted_return_5d)
        sent_deficit = max(0.0, s_sent - sentiment_score)
        ret_boost = min(0.18, (ret_deficit / 0.04) * 0.18)
        sent_boost = min(0.12, (sent_deficit / 0.40) * 0.12)
        vol_boost = min(0.05, max(0.0, (volatility - vol_tol) / 0.20) * 0.05)

        confidence = float(
            np.clip(0.68 + ret_boost + sent_boost + vol_boost, 0.50, 0.98)
        )
        reason = (
            f"Bearish signal ({canonical_profile} Risk): Predicted 5-day drop "
            f"({predicted_return_5d:+.2%}) breached the stop hurdle ({s_ret:+.2%}) "
            f"alongside deteriorating sentiment ({sentiment_score:+.2f} < {s_sent:+.2f}); exit advised."
        )

    else:
        recommendation = "Hold"

        # Diagnostic rationale for Hold
        if is_buy_candidate and is_volatility_exceeded:
            confidence = 0.82
            reason = (
                f"Capital preservation hold ({canonical_profile} Risk): Forecasted return "
                f"({predicted_return_5d:+.2%}) meets buy criteria, but elevated volatility "
                f"({volatility:.1%} > {vol_tol:.1%}) violates conservative risk limits."
            )
        elif predicted_return_5d > b_ret and sentiment_score <= b_sent:
            confidence = 0.72
            reason = (
                f"Hold advised: Positive return forecast ({predicted_return_5d:+.2%}) is unconfirmed "
                f"by news sentiment ({sentiment_score:+.2f} <= {b_sent:+.2f}); awaiting clearer market consensus."
            )
        elif predicted_return_5d <= s_ret and sentiment_score >= s_sent:
            confidence = 0.70
            reason = (
                f"Hold advised: Weak return projection ({predicted_return_5d:+.2%}) contrasts with "
                f"neutral/supportive sentiment ({sentiment_score:+.2f} >= {s_sent:+.2f}); avoiding premature exit."
            )
        else:
            confidence = 0.75
            reason = (
                f"Neutral market equilibrium ({canonical_profile} Risk): Predicted 5-day return "
                f"({predicted_return_5d:+.2%}) and news sentiment ({sentiment_score:+.2f}) remain "
                f"within the inactive buffer zone ({s_ret:+.2%} to {b_ret:+.2%})."
            )

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 4),
        "reason": reason,
        "risk_profile": canonical_profile,
        "metrics": {
            "predicted_return_5d": float(predicted_return_5d),
            "sentiment_score": float(sentiment_score),
            "volatility": float(volatility),
        },
        "thresholds": thresholds,
    }


def compute_performance_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Calculate essential portfolio performance metrics from daily return series:
    Total Return, CAGR, Annual Volatility, Sharpe Ratio, and Maximum Drawdown.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily percentage returns.
    risk_free_rate : float
        Annualized risk-free rate benchmark. Defaults to 0.06 (6.0%).

    Returns
    -------
    Dict[str, float]
        Dictionary with Total_Return, CAGR, Volatility, Sharpe_Ratio, and Max_Drawdown.
    """
    clean_returns = daily_returns.dropna()
    num_days = len(clean_returns)

    if num_days == 0:
        return {
            "Total_Return": 0.0,
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe_Ratio": 0.0,
            "Max_Drawdown": 0.0,
        }

    # Cumulative equity curve
    equity_curve = (1.0 + clean_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0)

    # Compound Annual Growth Rate (CAGR)
    years = max(num_days / 252.0, 1.0 / 252.0)
    cagr = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if total_return > -1.0
        else -1.0
    )

    # Annualized Volatility
    daily_std = float(clean_returns.std(ddof=1))
    annual_vol = daily_std * np.sqrt(252) if daily_std > 0 else 0.0

    # Annualized Sharpe Ratio
    daily_rf = risk_free_rate / 252.0
    excess_returns = clean_returns - daily_rf
    mean_excess = float(excess_returns.mean())
    sharpe_ratio = (mean_excess / daily_std) * np.sqrt(252) if daily_std > 1e-8 else 0.0

    # Maximum Drawdown (Peak-to-Trough)
    rolling_peak = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_peak) / rolling_peak
    max_drawdown = float(drawdown_series.min())

    return {
        "Total_Return": round(total_return, 4),
        "CAGR": round(cagr, 4),
        "Volatility": round(annual_vol, 4),
        "Sharpe_Ratio": round(sharpe_ratio, 4),
        "Max_Drawdown": round(max_drawdown, 4),
    }


def backtest_strategy(
    price_series: pd.Series,
    predicted_returns_5d: pd.Series,
    sentiment_scores: Optional[pd.Series] = None,
    volatility_series: Optional[pd.Series] = None,
    risk_profile: str = "Medium",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Execute a walk-forward, lookahead-free backtest comparing the algorithmic
    recommendation strategy against a passive Buy-and-Hold benchmark.

    Execution Framework (Rule 7 compliant):
      - Signal at date t is computed from data available up to date t.
      - Position determined at date t is held across date t+1.
      - When position is 0 (Cash / Flat), capital earns daily risk_free_rate.
      - When position is 1 (Invested), capital earns asset daily return.

    Parameters
    ----------
    price_series : pd.Series
        Historical closing prices with DateTimeIndex.
    predicted_returns_5d : pd.Series
        Predicted 5-day forward return series aligned with price_series.
    sentiment_scores : Optional[pd.Series]
        News sentiment scores in [-1, 1]. Defaults to neutral 0.0 if omitted.
    volatility_series : Optional[pd.Series]
        Rolling volatility series. Defaults to 20-day annualized rolling std.
    risk_profile : str
        'Low', 'Medium', or 'High'. Defaults to 'Medium'.
    risk_free_rate : float
        Annual risk-free rate. Defaults to 0.06.
    initial_capital : float
        Initial investment amount. Defaults to 100,000.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'metrics_comparison', 'equity_curve_df', 'summary', and 'trades'.
    """
    # Align dates across series
    common_idx = price_series.index.intersection(predicted_returns_5d.index)
    if len(common_idx) < 10:
        raise ValueError(
            f"Insufficient aligned trading days ({len(common_idx)}). Minimum 10 required."
        )

    prices = price_series.loc[common_idx].sort_index()
    pred_5d = predicted_returns_5d.loc[common_idx]

    # Handle sentiment series
    if sentiment_scores is None:
        sent = pd.Series(0.0, index=common_idx)
    else:
        sent = sentiment_scores.reindex(common_idx).fillna(0.0)

    # Handle volatility series
    asset_daily_returns = prices.pct_change().fillna(0.0)
    if volatility_series is None:
        vol = asset_daily_returns.rolling(20).std().bfill().fillna(0.20) * np.sqrt(252)
    else:
        vol = volatility_series.reindex(common_idx).bfill().fillna(0.20)

    # Walk-forward simulation
    dates = list(common_idx)
    signals: List[str] = []
    confidences: List[float] = []
    positions: List[float] = []

    current_position = 0.0  # Start in cash until first Buy signal
    trade_count = 0

    for d in dates:
        p_ret = float(pred_5d.loc[d])
        s_val = float(sent.loc[d])
        v_val = float(vol.loc[d])

        rec = generate_recommendation(
            predicted_return_5d=p_ret,
            sentiment_score=s_val,
            volatility=v_val,
            risk_profile=risk_profile,
        )
        sig = rec["recommendation"]
        conf = rec["confidence"]

        # Signal-to-Position Mapping:
        # Buy -> 1.0 (Long)
        # Sell -> 0.0 (Cash)
        # Hold -> Maintain prior position
        if sig == "Buy":
            new_position = 1.0
        elif sig == "Sell":
            new_position = 0.0
        else:
            new_position = current_position

        if new_position != current_position:
            trade_count += 1
            current_position = new_position

        signals.append(sig)
        confidences.append(conf)
        positions.append(current_position)

    # Shift position series by 1 day to ensure NO lookahead bias (Rule 7)
    # The trade decided at close of day t is held over day t+1
    pos_series = pd.Series(positions, index=common_idx).shift(1).fillna(0.0)

    daily_rf = risk_free_rate / 252.0
    strategy_daily_ret = (
        pos_series * asset_daily_returns + (1.0 - pos_series) * daily_rf
    )
    buy_hold_daily_ret = asset_daily_returns

    # Build equity curves
    strat_equity = initial_capital * (1.0 + strategy_daily_ret).cumprod()
    bh_equity = initial_capital * (1.0 + buy_hold_daily_ret).cumprod()

    # Drawdown series
    strat_peak = strat_equity.cummax()
    strat_dd = (strat_equity - strat_peak) / strat_peak

    bh_peak = bh_equity.cummax()
    bh_dd = (bh_equity - bh_peak) / bh_peak

    # Compute comparative metrics
    strat_metrics = compute_performance_metrics(
        strategy_daily_ret, risk_free_rate=risk_free_rate
    )
    bh_metrics = compute_performance_metrics(
        buy_hold_daily_ret, risk_free_rate=risk_free_rate
    )

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Total Return (%)",
                "CAGR (%)",
                "Annual Volatility (%)",
                "Sharpe Ratio",
                "Max Drawdown (%)",
            ],
            "Strategy": [
                round(strat_metrics["Total_Return"] * 100.0, 2),
                round(strat_metrics["CAGR"] * 100.0, 2),
                round(strat_metrics["Volatility"] * 100.0, 2),
                round(strat_metrics["Sharpe_Ratio"], 2),
                round(strat_metrics["Max_Drawdown"] * 100.0, 2),
            ],
            "Buy_and_Hold": [
                round(bh_metrics["Total_Return"] * 100.0, 2),
                round(bh_metrics["CAGR"] * 100.0, 2),
                round(bh_metrics["Volatility"] * 100.0, 2),
                round(bh_metrics["Sharpe_Ratio"], 2),
                round(bh_metrics["Max_Drawdown"] * 100.0, 2),
            ],
            "Outperformance": [
                round(
                    (strat_metrics["Total_Return"] - bh_metrics["Total_Return"])
                    * 100.0,
                    2,
                ),
                round((strat_metrics["CAGR"] - bh_metrics["CAGR"]) * 100.0, 2),
                round(
                    (strat_metrics["Volatility"] - bh_metrics["Volatility"]) * 100.0, 2
                ),
                round(strat_metrics["Sharpe_Ratio"] - bh_metrics["Sharpe_Ratio"], 2),
                round(
                    (strat_metrics["Max_Drawdown"] - bh_metrics["Max_Drawdown"])
                    * 100.0,
                    2,
                ),
            ],
        }
    )

    equity_df = pd.DataFrame(
        {
            "Price": prices,
            "Signal": signals,
            "Confidence": confidences,
            "Position": positions,
            "Strategy_Return": strategy_daily_ret,
            "Buy_Hold_Return": buy_hold_daily_ret,
            "Strategy_Equity": strat_equity,
            "Buy_Hold_Equity": bh_equity,
            "Strategy_Drawdown": strat_dd,
            "Buy_Hold_Drawdown": bh_dd,
        },
        index=common_idx,
    )

    return {
        "metrics_comparison": comparison_df,
        "strategy_metrics": strat_metrics,
        "buy_and_hold_metrics": bh_metrics,
        "equity_curve": equity_df,
        "trade_count": trade_count,
        "risk_profile": risk_profile,
        "risk_free_rate": risk_free_rate,
        "initial_capital": initial_capital,
    }


def plot_backtest_results(
    backtest_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Generate and persist a dual-panel visual performance chart comparing
    the Strategy Equity Curve and Drawdown against the Buy-and-Hold benchmark.

    Parameters
    ----------
    backtest_results : Dict[str, Any]
        Dictionary returned by backtest_strategy.
    output_path : Optional[Union[str, Path]]
        Destination path for image. Defaults to data/processed/backtest_results.png.
    title : Optional[str]
        Plot super-title.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    dest_path = (
        Path(output_path)
        if output_path
        else PROCESSED_DATA_DIR / "backtest_results.png"
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    equity_df = backtest_results["equity_curve"]
    strat_m = backtest_results["strategy_metrics"]
    bh_m = backtest_results["buy_and_hold_metrics"]
    profile = backtest_results.get("risk_profile", "Medium")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # Panel 1: Cumulative Equity Curve
    ax1.plot(
        equity_df.index,
        equity_df["Strategy_Equity"],
        label=f"Recommendation Strategy ({profile}): Ret {strat_m['Total_Return']:+.1%}, Sharpe {strat_m['Sharpe_Ratio']:.2f}",
        color="#1f77b4",
        linewidth=2.0,
    )
    ax1.plot(
        equity_df.index,
        equity_df["Buy_Hold_Equity"],
        label=f"Buy & Hold Benchmark: Ret {bh_m['Total_Return']:+.1%}, Sharpe {bh_m['Sharpe_Ratio']:.2f}",
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.8,
    )

    # Highlight Buy and Sell execution events
    buys = equity_df[equity_df["Signal"] == "Buy"]
    sells = equity_df[equity_df["Signal"] == "Sell"]

    if not buys.empty:
        ax1.scatter(
            buys.index,
            buys["Strategy_Equity"],
            marker="^",
            color="green",
            s=40,
            alpha=0.6,
            label="Buy Signal",
            zorder=5,
        )
    if not sells.empty:
        ax1.scatter(
            sells.index,
            sells["Strategy_Equity"],
            marker="v",
            color="red",
            s=40,
            alpha=0.6,
            label="Sell Signal",
            zorder=5,
        )

    ax1.set_ylabel("Portfolio Value (INR)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.set_title(
        title
        or f"Backtest Performance: Algorithmic Strategy vs Buy & Hold ({profile} Risk)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Panel 2: Underwater Drawdown Comparison
    ax2.plot(
        equity_df.index,
        equity_df["Strategy_Drawdown"] * 100.0,
        label=f"Strategy Drawdown (Max: {strat_m['Max_Drawdown']:.1%})",
        color="#1f77b4",
        linewidth=1.5,
    )
    ax2.plot(
        equity_df.index,
        equity_df["Buy_Hold_Drawdown"] * 100.0,
        label=f"Buy & Hold Drawdown (Max: {bh_m['Max_Drawdown']:.1%})",
        color="#d62728",
        linestyle=":",
        linewidth=1.4,
    )
    ax2.fill_between(
        equity_df.index,
        equity_df["Strategy_Drawdown"] * 100.0,
        0,
        color="#1f77b4",
        alpha=0.15,
    )
    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(dest_path, dpi=300)
    plt.close(fig)

    return dest_path


def run_backtest_pipeline(
    ticker: str = "RELIANCE.NS",
    risk_profile: str = "Medium",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Any]:
    """
    Automated backtesting pipeline for a specific stock ticker.
    Loads raw prices, derives 5-day predictive momentum forecasts and FinBERT sentiment,
    executes walk-forward backtest, saves comparison CSV and visualization PNG.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol. Defaults to 'RELIANCE.NS'.
    risk_profile : str
        'Low', 'Medium', or 'High'. Defaults to 'Medium'.
    risk_free_rate : float
        Benchmark risk-free rate.

    Returns
    -------
    Dict[str, Any]
        Full backtest results dictionary.
    """
    df_raw = load_raw_data(ticker)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")]
    close = df_raw["Close"].astype(float)
    returns = close.pct_change().dropna()

    # 5-day predictive momentum return shifted by 1 day (strictly historical, zero lookahead)
    pred_5d = close.pct_change(5).shift(1).dropna()
    aligned_idx = pred_5d.index.intersection(returns.index)

    close_aligned = close.loc[aligned_idx]
    pred_5d_aligned = pred_5d.loc[aligned_idx]
    vol_20d = returns.loc[aligned_idx].rolling(20).std().bfill().fillna(0.20) * np.sqrt(
        252
    )

    # Ingest FinBERT daily sentiment if available
    sent_series = pd.Series(0.0, index=aligned_idx)
    sent_csv = PROCESSED_DATA_DIR / "sentiment.csv"
    if sent_csv.exists():
        try:
            df_sent = pd.read_csv(sent_csv, parse_dates=["Date"])
            stock_sent = df_sent[df_sent["Ticker"] == ticker].set_index("Date")[
                "Daily_Sentiment"
            ]
            for d, val in stock_sent.items():
                if d in sent_series.index:
                    sent_series.loc[d] = float(val)
        except Exception:
            pass

    backtest_out = backtest_strategy(
        price_series=close_aligned,
        predicted_returns_5d=pred_5d_aligned,
        sentiment_scores=sent_series,
        volatility_series=vol_20d,
        risk_profile=risk_profile,
        risk_free_rate=risk_free_rate,
    )

    # Export comparison table to CSV
    csv_path = PROCESSED_DATA_DIR / "backtest_results.csv"
    backtest_out["metrics_comparison"].to_csv(csv_path, index=False)

    # Generate and export visualization
    plot_backtest_results(
        backtest_out,
        title=f"Backtest: {ticker} Algorithmic Strategy vs Buy & Hold ({risk_profile} Risk)",
    )

    return backtest_out


def main() -> None:
    """CLI execution entrypoint for recommender scoring and backtest."""
    print("=" * 70)
    print("Northgate AI Stock Predictor - Phase 6: Recommendation Engine & Backtest")
    print("=" * 70)

    print("\n[1/3] Testing Rule-Based Recommendation Scoring across Risk Profiles...")
    test_cases = [
        {"desc": "Strong Bullish Signal", "ret": 0.035, "sent": 0.25, "vol": 0.18},
        {"desc": "Mild Bullish Signal", "ret": 0.022, "sent": 0.12, "vol": 0.20},
        {"desc": "Bearish Breakdown", "ret": -0.028, "sent": -0.18, "vol": 0.24},
        {"desc": "Conflicting Divergence", "ret": 0.025, "sent": -0.15, "vol": 0.22},
        {"desc": "Neutral Chop", "ret": 0.005, "sent": 0.02, "vol": 0.16},
    ]

    for profile in ["Low", "Medium", "High"]:
        print(f"\n--- Risk Profile: {profile} ---")
        for case in test_cases:
            rec = generate_recommendation(
                predicted_return_5d=case["ret"],
                sentiment_score=case["sent"],
                volatility=case["vol"],
                risk_profile=profile,
            )
            print(
                f"  [{case['desc']:22s}] -> Action: {rec['recommendation']:4s} | "
                f"Confidence: {rec['confidence']:.1%} | {rec['reason']}"
            )

    print("\n[2/3] Executing Walk-Forward Backtest on RELIANCE.NS...")
    results = run_backtest_pipeline(ticker="RELIANCE.NS", risk_profile="Medium")

    print("\n[3/3] Backtest Performance Comparison (Strategy vs Buy-and-Hold):")
    print(results["metrics_comparison"].to_string(index=False))
    print(f"\nTrade Count (Position Transitions): {results['trade_count']}")
    print("Artifacts saved:")
    print(f"  - Table: {PROCESSED_DATA_DIR / 'backtest_results.csv'}")
    print(f"  - Chart: {PROCESSED_DATA_DIR / 'backtest_results.png'}")

    print("\n" + "=" * 70)
    print("Phase 6 recommender engine and backtest completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
