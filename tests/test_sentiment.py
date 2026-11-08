"""Unit tests for Phase 4: FinBERT news sentiment analysis and feature integration."""

import unittest

import numpy as np
import pandas as pd

from src.sentiment import (
    FinBERTSentimentAnalyzer,
    get_headlines_for_ticker,
    merge_sentiment_features,
)


class TestSentimentAnalysis(unittest.TestCase):
    """Test suite for sentiment scoring, news fetching, and feature merging."""

    @classmethod
    def setUpClass(cls):
        """Load the cached FinBERT analyzer once for testing."""
        cls.analyzer = FinBERTSentimentAnalyzer()

    def test_finbert_score_polarity(self):
        """Verify that positive and negative financial headlines receive expected polarity signs."""
        pos_headline = (
            "Company reports record-breaking quarterly net profit and surges 15%."
        )
        neg_headline = (
            "Severe regulatory penalties and major accounting fraud uncovered at firm."
        )

        pos_res = self.analyzer.score_headline(pos_headline)
        neg_res = self.analyzer.score_headline(neg_headline)

        # Polarity score must be strictly in [-1.0, 1.0]
        self.assertTrue(-1.0 <= pos_res["score"] <= 1.0)
        self.assertTrue(-1.0 <= neg_res["score"] <= 1.0)

        # Expected directionality
        self.assertGreater(
            pos_res["score"], 0.0, "Positive headline should have positive score."
        )
        self.assertLess(
            neg_res["score"], 0.0, "Negative headline should have negative score."
        )
        self.assertEqual(pos_res["label"], "positive")
        self.assertEqual(neg_res["label"], "negative")

    def test_get_headlines_for_ticker(self):
        """Verify headlines retrieval returns non-empty list with correct dict keys."""
        headlines = get_headlines_for_ticker("RELIANCE.NS")
        self.assertIsInstance(headlines, list)
        self.assertGreater(len(headlines), 0)
        for h in headlines:
            self.assertIn("headline", h)
            self.assertIn("date", h)
            self.assertTrue(len(h["headline"]) > 0)

    def test_merge_sentiment_features(self):
        """Verify sentiment scores merge smoothly into feature matrices with lag features."""
        dates = pd.date_range("2026-10-01", periods=10, freq="B")
        feature_df = pd.DataFrame(
            {"Close": np.linspace(100, 110, 10), "Volume": np.ones(10) * 1000},
            index=dates,
        )

        sentiment_df = pd.DataFrame(
            [
                {
                    "Date": "2026-10-02",
                    "Ticker": "RELIANCE.NS",
                    "Daily_Sentiment": 0.75,
                    "Headline_Count": 2,
                },
                {
                    "Date": "2026-10-05",
                    "Ticker": "RELIANCE.NS",
                    "Daily_Sentiment": -0.50,
                    "Headline_Count": 1,
                },
            ]
        )

        merged = merge_sentiment_features(
            feature_df, sentiment_df, ticker="RELIANCE.NS"
        )

        self.assertIn("Sentiment_Score", merged.columns)
        self.assertIn("Sentiment_Lag_1", merged.columns)
        self.assertEqual(len(merged), len(feature_df))
        self.assertEqual(
            merged.isna().sum().sum(), 0, "No NaNs allowed after sentiment merge."
        )


if __name__ == "__main__":
    unittest.main()
