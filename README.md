# Northgate AI Stock Predictor

An enterprise-grade, end-to-end AI platform for stock price forecasting (ML + DL), financial news sentiment analysis (FinBERT), modern portfolio optimization, and automated Buy/Hold/Sell decision intelligence, served via an interactive Streamlit dashboard.

---

## 🏗️ Architecture Roadmap & Phases

- [x] **Phase 1: Project Setup, Environment & Chronological Data Ingestion**
  - Modular project architecture (`src/`, `tests/`, `.env.example`, `.gitignore`).
  - High-resilience stock data fetcher via `yfinance` with automated caching (`data/cache/*.parquet`).
  - Strict chronological time-series splitting (`train_val_test_split`, `walk_forward_splits`) guaranteeing **zero lookahead bias** (Rule 7).
  - Automated verification suite & unit tests (`tests/test_data_pipeline.py`).
- [ ] **Phase 2: Technical Indicators & Feature Engineering**
  - Trend, momentum, volatility, and volume indicators (RSI, MACD, Bollinger Bands, ATR, Moving Averages).
  - Target creation and stationary transformations.
- [ ] **Phase 3: Machine Learning Forecasting Models**
  - Baseline Regressors / Classifiers (Linear Models, Random Forest, XGBoost).
  - Time-series walk-forward cross validation and backtesting metrics (MAE, RMSE, MAPE, Directional Accuracy).
- [ ] **Phase 4: Deep Learning Time-Series Models**
  - Sequential neural network modeling (LSTM / GRU architectures).
  - Multi-step horizon price forecasting.
- [ ] **Phase 5: Financial News Sentiment Analysis (FinBERT)**
  - Financial news headlines ingestion.
  - Transformer-based sentiment scoring (`ProsusAI/finbert`).
- [ ] **Phase 6: Portfolio Optimization & Recommendation Engine**
  - Mean-Variance / Modern Portfolio Theory optimization via PyPortfolioOpt.
  - Sentiment-tilted Sharpe ratio maximization and multi-factor Buy/Hold/Sell signal generation.
- [ ] **Phase 7: Interactive Streamlit Dashboard**
  - Multi-page interactive UI with Plotly charts, AI predictions, sentiment gauges, and allocation simulator.
- [ ] **Phase 8: End-to-End Integration, Stress Testing & Polish**

---

## 🚀 Quick Start (Phase 1)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/sahumohnish07-beep/Northgate-AI-Stock-Predictor.git
cd Northgate-AI-Stock-Predictor
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```
*(No API keys are committed or required for Phase 1).*

### 3. Verification & Tests
Run the automated test suite:
```bash
python -m unittest tests/test_data_pipeline.py
```

Run the interactive pipeline verification script:
```bash
python src/data/verify_pipeline.py AAPL
```

---

## 🔒 Time-Series Integrity (Rule 7)
Financial time series require strict temporal isolation:
- Data is partitioned chronologically into:
  - **Train:** $70\%$ (earliest window)
  - **Validation:** $15\%$ (intermediate window)
  - **Test:** $15\%$ (most recent window)
- Random shuffling is strictly prohibited to eliminate lookahead bias and data leakage.
