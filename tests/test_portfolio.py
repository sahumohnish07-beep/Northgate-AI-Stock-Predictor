"""
Unit tests for Portfolio Optimizer (Phase 5).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio import (
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKERS,
    compute_expected_returns_and_covariance,
    export_portfolio_allocations,
    generate_recommendations,
    load_historical_prices,
    optimize_all_profiles,
    optimize_portfolio,
    plot_efficient_frontier_and_profiles,
)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test suite covering data loading, return/risk calculations, and portfolio optimization."""

    @classmethod
    def setUpClass(cls):
        """Load market prices for default tickers once for all tests."""
        cls.prices = load_historical_prices(DEFAULT_TICKERS)
        cls.mu, cls.cov = compute_expected_returns_and_covariance(cls.prices)

    def test_load_historical_prices_structure(self):
        """Test that loaded prices DataFrame has correct shape, columns, and index."""
        self.assertFalse(self.prices.empty)
        self.assertGreaterEqual(len(self.prices), 100)
        for ticker in DEFAULT_TICKERS:
            self.assertIn(ticker, self.prices.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(self.prices[ticker]))
        self.assertTrue(isinstance(self.prices.index, pd.DatetimeIndex))

    def test_load_historical_prices_nonexistent_ticker(self):
        """Test that requesting a missing ticker raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_historical_prices(["NON_EXISTENT_TICKER_XYZ.NS"])

    def test_expected_returns_and_covariance_properties(self):
        """Test statistical properties of expected returns and covariance matrix."""
        # Check return vector dimensions
        self.assertEqual(len(self.mu), len(DEFAULT_TICKERS))
        self.assertEqual(self.cov.shape, (len(DEFAULT_TICKERS), len(DEFAULT_TICKERS)))

        # Covariance matrix must be symmetric
        np.testing.assert_allclose(
            self.cov.values,
            self.cov.values.T,
            atol=1e-8,
            err_msg="Covariance matrix is not symmetric",
        )

        # Covariance matrix must be positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(self.cov.values)
        self.assertTrue(
            all(val >= -1e-6 for val in eigenvalues),
            f"Negative eigenvalues found: {eigenvalues}",
        )

    def test_low_risk_optimization(self):
        """Test Low risk profile (minimum volatility)."""
        result = optimize_portfolio(
            mu=self.mu,
            cov_matrix=self.cov,
            risk_profile="Low",
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
        )
        self.assertIn("Low", result["risk_profile"])
        weights = result["weights"]

        # Weights must sum to 1.0 (within precision)
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

        # Long-only constraint (all weights >= 0)
        for ticker, w in weights.items():
            self.assertGreaterEqual(w, -1e-4)
            self.assertLessEqual(w, 1.0 + 1e-4)

        # Volatility must be positive
        self.assertGreater(result["annual_volatility"], 0.0)

    def test_medium_risk_optimization(self):
        """Test Medium risk profile (maximum Sharpe ratio)."""
        result = optimize_portfolio(
            mu=self.mu,
            cov_matrix=self.cov,
            risk_profile="Medium",
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
        )
        self.assertIn("Medium", result["risk_profile"])
        weights = result["weights"]

        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

        for ticker, w in weights.items():
            self.assertGreaterEqual(w, -1e-4)
            self.assertLessEqual(w, 1.0 + 1e-4)

        # Sharpe ratio must be a valid float
        self.assertIsInstance(result["sharpe_ratio"], float)

    def test_high_risk_optimization_weight_cap(self):
        """Test High risk profile enforces single-asset weight cap."""
        weight_cap = 0.40
        result = optimize_portfolio(
            mu=self.mu,
            cov_matrix=self.cov,
            risk_profile="High",
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
            weight_cap=weight_cap,
        )
        self.assertIn("High", result["risk_profile"])
        weights = result["weights"]

        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

        # Strictly check that no single weight exceeds the cap (allowing small rounding tolerance)
        for ticker, w in weights.items():
            self.assertLessEqual(
                w,
                weight_cap + 1e-3,
                f"{ticker} weight {w} exceeded weight_cap {weight_cap}",
            )
            self.assertGreaterEqual(w, -1e-4)

    def test_invalid_risk_profile(self):
        """Test that invalid risk profile names raise ValueError."""
        with self.assertRaises(ValueError):
            optimize_portfolio(
                mu=self.mu, cov_matrix=self.cov, risk_profile="InvalidProfile"
            )

    def test_invalid_weight_cap_infeasible(self):
        """Test that setting weight_cap < 1/N raises ValueError."""
        # For 5 assets, 1/5 = 0.20. Setting cap to 0.10 is mathematically infeasible
        with self.assertRaises(ValueError):
            optimize_portfolio(
                mu=self.mu, cov_matrix=self.cov, risk_profile="High", weight_cap=0.10
            )

    def test_optimize_all_profiles(self):
        """Test optimizing all three profiles in a single pass."""
        all_results = optimize_all_profiles(
            mu=self.mu,
            cov_matrix=self.cov,
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
        )
        self.assertEqual(set(all_results.keys()), {"Low", "Medium", "High"})
        # Low risk volatility should be <= High risk volatility
        self.assertLessEqual(
            all_results["Low"]["annual_volatility"],
            all_results["High"]["annual_volatility"] + 1e-2,
        )

    def test_export_portfolio_allocations(self):
        """Test exporting portfolio allocation table to CSV."""
        all_results = optimize_all_profiles(mu=self.mu, cov_matrix=self.cov)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_allocations.csv"
            df = export_portfolio_allocations(all_results, output_path=tmp_path)

            self.assertTrue(tmp_path.exists())
            self.assertEqual(len(df), 3)  # 3 profiles
            self.assertIn("Expected_Return", df.columns)
            self.assertIn("Sharpe_Ratio", df.columns)
            for ticker in DEFAULT_TICKERS:
                self.assertIn(f"Weight_{ticker}", df.columns)

    def test_generate_recommendations(self):
        """Test algorithmic recommendation generator."""
        all_results = optimize_all_profiles(mu=self.mu, cov_matrix=self.cov)
        rec_df = generate_recommendations(all_results)

        self.assertEqual(len(rec_df), len(DEFAULT_TICKERS))
        self.assertIn("Ticker", rec_df.columns)
        self.assertIn("Recommendation", rec_df.columns)
        self.assertIn("Mean_Weight", rec_df.columns)

    def test_plot_efficient_frontier_generation(self):
        """Test generating and saving efficient frontier PNG."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_plot = Path(tmp_dir) / "test_frontier.png"
            saved_path = plot_efficient_frontier_and_profiles(
                mu=self.mu,
                cov_matrix=self.cov,
                output_path=tmp_plot,
            )
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)  # File should be > 1KB


if __name__ == "__main__":
    unittest.main()
