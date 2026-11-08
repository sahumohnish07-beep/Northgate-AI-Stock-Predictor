"""Financial news sentiment analysis module using FinBERT (ProsusAI/finbert).

Fetches stock news headlines via Google News RSS (feedparser) or NewsAPI,
scores each headline as positive/neutral/negative, maps to a [-1, +1] polarity score,
aggregates daily sentiment per stock, merges into technical features,
and retrains the best forecasting model to measure sentiment impact.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import sys
import urllib.parse
from dotenv import load_dotenv

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

import feedparser
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import DEFAULT_TICKERS, load_raw_data
from src.features import engineer_features
from src.models import compute_metrics, split_time_series_80_20

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentiment")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_CSV_PATH = PROCESSED_DIR / "sentiment.csv"
COMPARISON_CSV_PATH = PROCESSED_DIR / "sentiment_model_comparison.csv"

FINBERT_MODEL_NAME = "ProsusAI/finbert"


class FinBERTSentimentAnalyzer:
    """Encapsulates Hugging Face ProsusAI/finbert for financial sentiment scoring."""

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: Optional[torch.device] = None) -> None:
        """Initializes FinBERT tokenizer and pre-trained classification model.

        Args:
            model_name: Hugging Face hub model identifier.
            device: Torch execution device (cpu or cuda).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        self.id2label = self.model.config.id2label
        logger.info(f"FinBERT loaded successfully. Config labels: {self.id2label}")

    def score_headline(self, headline: str) -> Dict[str, Any]:
        """Scores a single headline and returns label probabilities and normalized score.

        Score formula: P(positive) - P(negative), bounded strictly between [-1.0, +1.0].

        Args:
            headline: Text string of the news headline.

        Returns:
            Dict containing label, positive, negative, neutral probs, and score in [-1.0, 1.0].
        """
        if not headline or not headline.strip():
            return {
                "headline": headline,
                "label": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

        inputs = self.tokenizer(
            headline,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        # Extract probabilities according to FinBERT id2label mapping
        prob_dict = {self.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        pos_prob = prob_dict.get("positive", 0.0)
        neg_prob = prob_dict.get("negative", 0.0)
        neu_prob = prob_dict.get("neutral", 0.0)

        # Polarity score: +1 (pure positive) to -1 (pure negative)
        score = pos_prob - neg_prob
        predicted_label = max(prob_dict.items(), key=lambda x: x[1])[0]

        return {
            "headline": headline,
            "label": predicted_label,
            "score": round(float(score), 4),
            "positive": round(pos_prob, 4),
            "negative": round(neg_prob, 4),
            "neutral": round(neu_prob, 4),
        }

    def score_headlines_batch(self, headlines: List[str]) -> List[Dict[str, Any]]:
        """Scores a list of headlines in batches.

        Args:
            headlines: List of headline strings.

        Returns:
            List of scoring dictionaries.
        """
        return [self.score_headline(h) for h in headlines]


# ---------------------------------------------------------------------------
# News Fetchers (Google News RSS, NewsAPI, and Fallback Sample Dataset)
# ---------------------------------------------------------------------------

FALLBACK_HEADLINES: Dict[str, List[Tuple[str, str]]] = {
    "RELIANCE.NS": [
        ("Reliance Industries posts record quarterly profit driven by retail and telecom growth.", "2026-10-25"),
        ("Jio expands 5G enterprise solutions, adding high-margin enterprise accounts.", "2026-10-28"),
        ("Oil refinery margins face minor headwinds amid fluctuating crude prices.", "2026-11-01"),
        ("Reliance Retail accelerates aggressive physical store expansion nationwide.", "2026-11-03"),
        ("Analysts maintain bullish buy rating on Reliance Industries on green energy momentum.", "2026-11-05"),
    ],
    "TCS.NS": [
        ("TCS secures multi-year billion-dollar digital transformation contract in Europe.", "2026-10-26"),
        ("Tech sector faces minor wage inflation and cautious IT discretionary spending.", "2026-10-30"),
        ("TCS expands generative AI deployment across banking and insurance clients.", "2026-11-02"),
        ("Q3 operating margins projected to expand following operational optimization.", "2026-11-04"),
    ],
    "INFY.NS": [
        ("Infosys raises annual revenue guidance after stellar quarterly deal wins.", "2026-10-27"),
        ("Infosys collaborates with global cloud providers for enterprise AI integration.", "2026-10-31"),
        ("Attrition rates at Infosys decline to lowest level in multiple quarters.", "2026-11-03"),
        ("Analysts express caution regarding delayed client decision cycles in North America.", "2026-11-05"),
    ],
    "HDFCBANK.NS": [
        ("HDFC Bank reports robust deposit mobilization and steady net interest margins.", "2026-10-24"),
        ("Credit card additions surge as retail spending rebounds strongly.", "2026-10-29"),
        ("Gross NPA ratios remain stable with excellent asset quality across loan books.", "2026-11-02"),
        ("RBI monetary policy keeps interest rates steady, benefiting private banks.", "2026-11-04"),
    ],
    "ITC.NS": [
        ("ITC cigarette and FMCG volumes show steady growth and pricing power.", "2026-10-27"),
        ("ITC hotels business demerger reaches final regulatory approval stages.", "2026-10-30"),
        ("Agri-business revenue surges on strong international commodity exports.", "2026-11-02"),
        ("ITC paperboards division experiences mild raw material input cost pressure.", "2026-11-05"),
    ],
}


def fetch_google_news_rss(query: str, max_articles: int = 15) -> List[Dict[str, str]]:
    """Fetches real-time financial headlines using Google News RSS.

    Args:
        query: Search term (e.g. 'Reliance Industries stock').
        max_articles: Maximum entries to retrieve.

    Returns:
        List of dicts with 'headline' and 'date' ('YYYY-MM-DD').
    """
    clean_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={clean_query}+when:30d&hl=en-IN&gl=IN&ceid=IN:en"
    logger.info(f"Querying Google News RSS for: {query}")

    try:
        feed = feedparser.parse(rss_url)
        articles: List[Dict[str, str]] = []

        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "")
            # Remove publisher suffix if present (e.g. "Headline - Economic Times")
            clean_title = title.rsplit(" - ", 1)[0].strip()

            # Parse date
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                date_str = datetime(*published_parsed[:6]).strftime("%Y-%m-%d")
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")

            if clean_title:
                articles.append({"headline": clean_title, "date": date_str})

        logger.info(f"Retrieved {len(articles)} articles from Google News RSS for {query}.")
        return articles
    except Exception as exc:
        logger.warning(f"Google News RSS fetch failed for {query}: {exc}")
        return []


def fetch_news_api(query: str, api_key: str, max_articles: int = 15) -> List[Dict[str, str]]:
    """Fetches articles from NewsAPI if an API key is available.

    Args:
        query: Search query.
        api_key: NewsAPI key.
        max_articles: Max articles to return.

    Returns:
        List of dicts with 'headline' and 'date'.
    """
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for art in data.get("articles", [])[:max_articles]:
                title = art.get("title", "")
                pub_date = art.get("publishedAt", "")[:10]
                if title:
                    articles.append({"headline": title, "date": pub_date})
            logger.info(f"Retrieved {len(articles)} articles from NewsAPI for {query}.")
            return articles
    except Exception as exc:
        logger.warning(f"NewsAPI query failed: {exc}")

    return []


def get_headlines_for_ticker(ticker: str) -> List[Dict[str, str]]:
    """Fetches headlines for a ticker attempting NewsAPI, then RSS, then sample fallback.

    Args:
        ticker: Ticker symbol (e.g. 'RELIANCE.NS').

    Returns:
        List of dicts with 'headline' and 'date'.
    """
    company_queries = {
        "RELIANCE.NS": "Reliance Industries stock",
        "TCS.NS": "Tata Consultancy Services TCS stock",
        "INFY.NS": "Infosys stock",
        "HDFCBANK.NS": "HDFC Bank stock",
        "ITC.NS": "ITC limited stock",
    }
    query = company_queries.get(ticker, ticker)

    # 1. Check NewsAPI from .env
    news_api_key = os.getenv("NEWS_API_KEY", "").strip()
    if news_api_key:
        articles = fetch_news_api(query, news_api_key)
        if articles:
            return articles

    # 2. Try Google News RSS
    rss_articles = fetch_google_news_rss(query)
    if rss_articles:
        return rss_articles

    # 3. Fallback sample dataset
    logger.info(f"Falling back to curated sample headlines for {ticker}.")
    fallback = FALLBACK_HEADLINES.get(ticker, [])
    return [{"headline": h, "date": d} for h, d in fallback]


# ---------------------------------------------------------------------------
# Daily Sentiment Aggregation & Feature Integration
# ---------------------------------------------------------------------------

def compute_daily_sentiment_dataset(
    tickers: Optional[List[str]] = None,
    analyzer: Optional[FinBERTSentimentAnalyzer] = None,
    output_path: Path = SENTIMENT_CSV_PATH,
) -> pd.DataFrame:
    """Fetches, scores, aggregates headlines per ticker per day, and saves to CSV.

    Args:
        tickers: List of stock symbols (default DEFAULT_TICKERS).
        analyzer: FinBERTSentimentAnalyzer instance.
        output_path: Destination CSV path.

    Returns:
        pd.DataFrame with columns ['Date', 'Ticker', 'Daily_Sentiment', 'Headline_Count'].
    """
    target_tickers = tickers or DEFAULT_TICKERS
    finbert = analyzer or FinBERTSentimentAnalyzer()

    all_scored_records: List[Dict[str, Any]] = []

    for ticker in target_tickers:
        logger.info(f"Processing news sentiment for {ticker}...")
        headlines = get_headlines_for_ticker(ticker)
        for item in headlines:
            score_res = finbert.score_headline(item["headline"])
            all_scored_records.append({
                "Date": item["date"],
                "Ticker": ticker,
                "Headline": item["headline"],
                "Label": score_res["label"],
                "Score": score_res["score"],
            })

    if not all_scored_records:
        logger.warning("No headlines retrieved. Returning empty DataFrame.")
        return pd.DataFrame()

    df_scored = pd.DataFrame(all_scored_records)
    # Aggregate daily sentiment score per stock
    daily_sentiment = (
        df_scored.groupby(["Date", "Ticker"])
        .agg(
            Daily_Sentiment=("Score", "mean"),
            Headline_Count=("Score", "count"),
        )
        .reset_index()
    )

    daily_sentiment["Daily_Sentiment"] = daily_sentiment["Daily_Sentiment"].round(4)
    daily_sentiment.to_csv(output_path, index=False)
    logger.info(f"Aggregated daily sentiment saved to {output_path} ({len(daily_sentiment)} records).")

    return daily_sentiment


def merge_sentiment_features(
    feature_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Merges daily sentiment scores into feature matrix with lag features to prevent lookahead bias.

    Args:
        feature_df: Feature DataFrame with DatetimeIndex from engineer_features().
        sentiment_df: Sentiment DataFrame with Date, Ticker, Daily_Sentiment.
        ticker: Stock symbol to filter.

    Returns:
        Augmented feature DataFrame with 'Sentiment_Score' and 'Sentiment_Lag_1'.
    """
    data = feature_df.copy()
    data["Date_Str"] = data.index.strftime("%Y-%m-%d")

    ticker_sentiment = sentiment_df[sentiment_df["Ticker"] == ticker].copy()

    if not ticker_sentiment.empty:
        merged = pd.merge(
            data,
            ticker_sentiment[["Date", "Daily_Sentiment"]],
            left_on="Date_Str",
            right_on="Date",
            how="left",
        )
        merged.index = data.index
        merged["Sentiment_Score"] = merged["Daily_Sentiment"].fillna(0.0)
    else:
        merged = data.copy()
        merged["Sentiment_Score"] = 0.0

    # Lag sentiment to guarantee zero lookahead bias
    merged["Sentiment_Lag_1"] = merged["Sentiment_Score"].shift(1).fillna(0.0)

    # Clean intermediate columns
    for col in ["Date_Str", "Date", "Daily_Sentiment"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    return merged


def retrain_with_and_without_sentiment(
    ticker: str = "RELIANCE.NS",
    sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Retrains the best model (Linear Regression) with and without FinBERT sentiment

    and generates a comparison table.

    Args:
        ticker: Stock symbol.
        sentiment_df: Optional precomputed sentiment DataFrame.

    Returns:
        pd.DataFrame containing side-by-side performance comparison.
    """
    logger.info(f"=== Retraining Best Model with vs without FinBERT Sentiment for {ticker} ===")
    raw_df = load_raw_data(ticker)
    base_feat_df = engineer_features(raw_df, include_target=True)

    if sentiment_df is None:
        if SENTIMENT_CSV_PATH.exists():
            sentiment_df = pd.read_csv(SENTIMENT_CSV_PATH)
        else:
            sentiment_df = compute_daily_sentiment_dataset([ticker])

    # 1. Baseline Model WITHOUT Sentiment
    X_train_base, y_train, X_test_base, y_test, _ = split_time_series_80_20(
        base_feat_df,
        target_col="Target_Next_Close",
        train_ratio=0.80,
    )

    model_no_sent = LinearRegression()
    model_no_sent.fit(X_train_base, y_train)
    preds_no_sent = model_no_sent.predict(X_test_base)
    metrics_no_sent = compute_metrics(y_test.values, preds_no_sent)

    # 2. Model WITH Sentiment
    sent_feat_df = merge_sentiment_features(base_feat_df, sentiment_df, ticker)
    X_train_sent, _, X_test_sent, _, _ = split_time_series_80_20(
        sent_feat_df,
        target_col="Target_Next_Close",
        train_ratio=0.80,
    )

    model_with_sent = LinearRegression()
    model_with_sent.fit(X_train_sent, y_train)
    preds_with_sent = model_with_sent.predict(X_test_sent)
    metrics_with_sent = compute_metrics(y_test.values, preds_with_sent)

    comparison_results = pd.DataFrame([
        {
            "Ticker": ticker,
            "Variant": "Without Sentiment (Baseline)",
            "Features": len(X_train_base.columns),
            "Test_RMSE": metrics_no_sent["RMSE"],
            "Test_MAE": metrics_no_sent["MAE"],
            "Test_MAPE_%": metrics_no_sent["MAPE_%"],
        },
        {
            "Ticker": ticker,
            "Variant": "With FinBERT Sentiment",
            "Features": len(X_train_sent.columns),
            "Test_RMSE": metrics_with_sent["RMSE"],
            "Test_MAE": metrics_with_sent["MAE"],
            "Test_MAPE_%": metrics_with_sent["MAPE_%"],
        },
    ])

    comparison_results.to_csv(COMPARISON_CSV_PATH, index=False)
    logger.info(f"Saved sentiment comparison table to {COMPARISON_CSV_PATH}")

    return comparison_results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    analyzer_instance = FinBERTSentimentAnalyzer()
    sent_df = compute_daily_sentiment_dataset(analyzer=analyzer_instance)
    comp_df = retrain_with_and_without_sentiment(ticker=target, sentiment_df=sent_df)

    print("\n" + "=" * 75)
    print("FINBERT DAILY SENTIMENT SAMPLE")
    print("=" * 75)
    print(sent_df.head(10).to_string(index=False))
    print("\n" + "=" * 75)
    print("MODEL RETRAINING: WITH vs WITHOUT SENTIMENT")
    print("=" * 75)
    print(comp_df.to_string(index=False))
    print("=" * 75)

