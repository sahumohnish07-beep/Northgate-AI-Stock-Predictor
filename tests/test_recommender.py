"""
Unit tests for Recommendation Engine and Backtesting (Phase 6).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommender import (
    DEFAULT_RISK_FREE_RATE,
    backtest_strategy,
    compute_performance_metrics,
    generate_recommendation,
    get_risk_thresholds,
    plot_backtest_results,
)


class TestRecommendationEngine(unittest.TestCase):
    """Test suite for recommendation scoring, risk thresholding, and backtesting."""

    def test_get_risk_thresholds_valid(self):
        """Test retrieving thresholds for standard profiles and aliases."""
        for profile in [
            "Low",
            "Medium",
            "High",
            "Conservative",
            "Balanced",
            "Aggressive",
        ]:
            th = get_risk_thresholds(profile)
            self.assertIn("buy_return", th)
            self.assertIn("buy_sentiment", th)
            self.assertIn("sell_return", th)
            self.assertIn("sell_sentiment", th)
            self.assertIn("volatility_tolerance", th)

        # Check relative threshold order
        th_low = get_risk_thresholds("Low")
        th_med = get_risk_thresholds("Medium")
        th_high = get_risk_thresholds("High")

        self.assertGreater(th_low["buy_return"], th_med["buy_return"])
        self.assertGreater(th_med["buy_return"], th_high["buy_return"])
        self.assertGreater(th_low["sell_return"], th_med["sell_return"])
        self.assertGreater(th_med["sell_return"], th_high["sell_return"])

    def test_get_risk_thresholds_invalid(self):
        """Test that invalid risk profiles raise ValueError."""
        with self.assertRaises(ValueError):
            get_risk_thresholds("UltraAggressiveSpeculation")

    def test_generate_recommendation_buy_medium(self):
        """Test Buy rule under Medium risk profile (> +2.0% return and > 0.10 sentiment)."""
        res = generate_recommendation(
            predicted_return_5d=0.035,
            sentiment_score=0.25,
            volatility=0.18,
            risk_profile="Medium",
        )
        self.assertEqual(res["recommendation"], "Buy")
        self.assertGreaterEqual(res["confidence"], 0.65)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("Bullish", res["reason"])
        self.assertIn("+3.50%", res["reason"])

    def test_generate_recommendation_sell_medium(self):
        """Test Sell rule under Medium risk profile (< -2.0% return and < -0.10 sentiment)."""
        res = generate_recommendation(
            predicted_return_5d=-0.030,
            sentiment_score=-0.20,
            volatility=0.22,
            risk_profile="Medium",
        )
        self.assertEqual(res["recommendation"], "Sell")
        self.assertGreaterEqual(res["confidence"], 0.65)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("Bearish", res["reason"])

    def test_generate_recommendation_hold_neutral(self):
        """Test Hold recommendation within neutral zone."""
        res = generate_recommendation(
            predicted_return_5d=0.005,
            sentiment_score=0.02,
            volatility=0.15,
            risk_profile="Medium",
        )
        self.assertEqual(res["recommendation"], "Hold")
        self.assertGreaterEqual(res["confidence"], 0.50)
        self.assertIn("equilibrium", res["reason"])

    def test_generate_recommendation_hold_conflicting_signals(self):
        """Test Hold recommendation when return forecast and sentiment diverge."""
        # Positive return, negative sentiment
        res_bull_bear = generate_recommendation(
            predicted_return_5d=0.030,
            sentiment_score=-0.15,
            volatility=0.18,
            risk_profile="Medium",
        )
        self.assertEqual(res_bull_bear["recommendation"], "Hold")
        self.assertIn("unconfirmed", res_bull_bear["reason"])

        # Negative return, positive sentiment
        res_bear_bull = generate_recommendation(
            predicted_return_5d=-0.030,
            sentiment_score=0.20,
            volatility=0.18,
            risk_profile="Medium",
        )
        self.assertEqual(res_bear_bull["recommendation"], "Hold")
        self.assertIn("contrasts", res_bear_bull["reason"])

    def test_risk_profile_threshold_adaptability(self):
        """Test that different profiles adapt Buy/Sell triggers for the same input."""
        # 1. Return = +2.5%, Sentiment = +0.12
        # Medium: > +2% and > 0.1 -> Buy
        # Low: requires > +3% and > 0.15 -> Hold
        res_med = generate_recommendation(0.025, 0.12, 0.20, risk_profile="Medium")
        res_low = generate_recommendation(0.025, 0.12, 0.20, risk_profile="Low")
        self.assertEqual(res_med["recommendation"], "Buy")
        self.assertEqual(res_low["recommendation"], "Hold")

        # 2. Return = +1.5%, Sentiment = +0.05
        # High: requires > +1% and > 0.0 -> Buy
        # Medium: requires > +2% and > 0.1 -> Hold
        res_high = generate_recommendation(0.015, 0.05, 0.20, risk_profile="High")
        res_med2 = generate_recommendation(0.015, 0.05, 0.20, risk_profile="Medium")
        self.assertEqual(res_high["recommendation"], "Buy")
        self.assertEqual(res_med2["recommendation"], "Hold")

        # 3. Return = -1.8%, Sentiment = -0.08
        # Low: triggers Sell (< -1.5% and < -0.05)
        # Medium: stays in Hold (< -2.0% required)
        res_low_sell = generate_recommendation(-0.018, -0.08, 0.20, risk_profile="Low")
        res_med_sell = generate_recommendation(
            -0.018, -0.08, 0.20, risk_profile="Medium"
        )
        self.assertEqual(res_low_sell["recommendation"], "Sell")
        self.assertEqual(res_med_sell["recommendation"], "Hold")

    def test_low_risk_high_volatility_restraint(self):
        """Test that extreme volatility prevents Low risk profile from entering a Buy."""
        # High return (+4%) and high sentiment (+0.30), but excessive volatility (40%)
        res = generate_recommendation(
            predicted_return_5d=0.040,
            sentiment_score=0.30,
            volatility=0.40,
            risk_profile="Low",
        )
        self.assertEqual(res["recommendation"], "Hold")
        self.assertIn("preservation", res["reason"])

    def test_compute_performance_metrics_synthetic(self):
        """Test performance metrics computation on synthetic returns."""
        # +1% per day for 100 days
        daily_ret = pd.Series([0.01] * 100)
        metrics = compute_performance_metrics(daily_ret, risk_free_rate=0.06)

        self.assertGreater(metrics["Total_Return"], 1.5)  # 1.01^100 - 1 = 1.7048
        self.assertEqual(
            metrics["Max_Drawdown"], 0.0
        )  # Strictly increasing, zero drawdown
        self.assertGreater(metrics["CAGR"], 0.0)

    def test_compute_performance_metrics_empty(self):
        """Test performance metrics on empty series."""
        metrics = compute_performance_metrics(pd.Series(dtype=float))
        self.assertEqual(metrics["Total_Return"], 0.0)
        self.assertEqual(metrics["Max_Drawdown"], 0.0)

    def test_backtest_strategy_execution(self):
        """Test walk-forward backtesting simulation with realistic synthetic time series."""
        dates = pd.date_range("2023-01-01", periods=150, freq="B")
        np.random.seed(42)
        sim_returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = pd.Series(1000.0 * np.cumprod(1.0 + sim_returns), index=dates)

        # Generate synthetic predicted returns and sentiments
        preds = pd.Series(np.random.normal(0.005, 0.025, len(dates)), index=dates)
        sents = pd.Series(np.random.normal(0.05, 0.30, len(dates)), index=dates)

        results = backtest_strategy(
            price_series=prices,
            predicted_returns_5d=preds,
            sentiment_scores=sents,
            risk_profile="Medium",
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
            initial_capital=50000.0,
        )

        self.assertIn("metrics_comparison", results)
        self.assertIn("strategy_metrics", results)
        self.assertIn("buy_and_hold_metrics", results)
        self.assertIn("equity_curve", results)

        comparison_df = results["metrics_comparison"]
        self.assertEqual(len(comparison_df), 5)
        self.assertIn("Total Return (%)", comparison_df["Metric"].values)
        self.assertIn("Max Drawdown (%)", comparison_df["Metric"].values)
        self.assertIn("Sharpe Ratio", comparison_df["Metric"].values)

        eq_df = results["equity_curve"]
        self.assertEqual(len(eq_df), 150)
        self.assertIn("Strategy_Equity", eq_df.columns)
        self.assertIn("Buy_Hold_Equity", eq_df.columns)
        self.assertIn("Signal", eq_df.columns)
        self.assertIn("Position", eq_df.columns)

        # Check drawdown is <= 0
        self.assertLessEqual(results["strategy_metrics"]["Max_Drawdown"], 1e-6)
        self.assertLessEqual(results["buy_and_hold_metrics"]["Max_Drawdown"], 1e-6)

    def test_plot_backtest_results_generation(self):
        """Test generating and saving backtest visualization PNG."""
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        prices = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        preds = pd.Series([0.025] * len(dates), index=dates)
        sents = pd.Series([0.20] * len(dates), index=dates)

        backtest_out = backtest_strategy(
            price_series=prices,
            predicted_returns_5d=preds,
            sentiment_scores=sents,
            risk_profile="Medium",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_plot = Path(tmp_dir) / "test_backtest.png"
            saved_path = plot_backtest_results(backtest_out, output_path=tmp_plot)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
