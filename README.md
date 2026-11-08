# Northgate AI Stock Predictor

An enterprise-grade, end-to-end AI platform that predicts stock prices using Machine Learning and Deep Learning, evaluates financial news sentiment using FinBERT, optimizes investment portfolios with Modern Portfolio Theory, and delivers algorithmic Buy/Hold/Sell recommendations through an interactive Streamlit dashboard.

---

> ⚠️ **IMPORTANT DISCLAIMER**  
> **This platform is for educational and research purposes only, and does not constitute financial or investment advice.** Always conduct your own due diligence before executing any trades or investments.

---

## 🏗️ Phase Roadmap

- [x] **Phase 0: Project Setup & Structure**
  - Standardized directory layout (`data/raw`, `data/processed`, `notebooks`, `src`, `models`, `tests`).
  - Core requirements, `.gitignore`, `.env.example`, and baseline Streamlit app.
- [x] **Phase 1: Data Collection & EDA**
  - Built `src/data_loader.py` with `yfinance` to fetch 5 years of daily OHLCV data.
  - Default coverage for premier Indian equities: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ITC.NS`.
  - Raw CSV persistence in `data/raw/<ticker>.csv` and automated EDA notebook (`notebooks/01_eda.ipynb`).
- [x] **Phase 2: Feature Engineering & Baseline ML Models**
  - Built `src/features.py`: RSI (14), MACD, SMA/EMA (10, 20, 50), Bollinger Bands, returns, rolling volatilities, and lag features.
  - Built `src/models.py`: Linear Regression, Random Forest, and XGBoost with chronological 80/20 train/test split (Rule 7).
  - Evaluated with RMSE, MAE, and MAPE; baseline table saved to `data/processed/baseline_results.csv`.
- [x] **Phase 3: Deep Learning (LSTM) Model**
  - Built `src/lstm_model.py` with PyTorch: stacked LSTM network with 60-day historical window.
  - Leak-free preprocessing: `MinMaxScaler` fitted **strictly on train set only**.
  - `EarlyStopping` and `ModelCheckpoint` callbacks to prevent overfitting.
  - Prediction plot saved to `data/processed/lstm_actual_vs_predicted.png` and model to `models/lstm_model.pt`.
- [x] **Phase 4: Financial News Sentiment Analysis (FinBERT)**
  - Built `src/sentiment.py` using Hugging Face `ProsusAI/finbert`.
  - Multi-source headlines fetcher supporting Google News RSS (`feedparser`), NewsAPI (key from `.env`), and curated fallbacks.
  - Scored each headline as positive, neutral, or negative and transformed into normalized polarity score $[-1.0, +1.0]$.
  - Aggregated daily sentiment scores per stock saved to `data/processed/sentiment.csv`.
  - Merged sentiment scores and lag features into the technical feature matrix without lookahead leakage.
  - Retrained the best model to evaluate predictive impact; results saved to `data/processed/sentiment_model_comparison.csv`.
  - Unit tests added in `tests/test_sentiment.py`.
- [ ] **Phase 5: Portfolio Optimization & Recommendation Engine**
  - PyPortfolioOpt Modern Portfolio Theory and multi-factor Buy/Hold/Sell decisions.
- [ ] **Phase 6: Interactive Streamlit Dashboard**
  - Plotly interactive charts, model comparison, and portfolio allocation UI.

---

## 📰 Phase 4 FinBERT Sentiment & Model Retraining Benchmark

Evaluated on **RELIANCE.NS** using chronological 80/20 train/test split:

| Model Variant | Feature Count | Test RMSE | Test MAE | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Without Sentiment (Baseline)** | 35 | 20.7664 | 14.8791 | 1.1080% |
| **With FinBERT Sentiment** | 37 | 20.7664 | 14.8791 | 1.1080% |

*Data Artifacts:*
- Daily aggregated sentiment dataset: `data/processed/sentiment.csv`
- Model comparison table: `data/processed/sentiment_model_comparison.csv`

---

## 🛠️ Quick Start

### 1. Download Market Data
```bash
python src/data_loader.py
```

### 2. Train & Evaluate Baseline Models
```bash
python src/models.py
```

### 3. Train & Evaluate 60-Day LSTM Model
```bash
python src/lstm_model.py RELIANCE.NS
```

### 4. Run FinBERT Sentiment Analysis & Retrain
```bash
python src/sentiment.py RELIANCE.NS
```

### 5. Run All Unit Tests
```bash
python -m unittest discover tests
```
