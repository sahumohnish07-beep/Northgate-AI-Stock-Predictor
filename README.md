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
  - Missing data cleaning, error handling for delisted/invalid tickers, and raw CSV persistence in `data/raw/<ticker>.csv`.
  - Comprehensive exploratory analysis notebook (`notebooks/01_eda.ipynb`) with price trajectories, return distributions, annualized volatilities, and correlation heatmaps.
  - Automated test suite in `tests/test_data_loader.py`.
- [x] **Phase 2: Feature Engineering & Baseline ML Models**
  - Built `src/features.py`: RSI (14), MACD (line, signal, histogram), SMA & EMA (10, 20, 50), Bollinger Bands (upper, middle, lower, width), daily returns, rolling volatilities (10d, 20d), and lag features (1, 2, 3, 5 periods).
  - Built `src/models.py`: Linear Regression, Random Forest, and XGBoost to predict next-day close.
  - Strict chronological time-based train/test split (80% train, 20% test) with zero lookahead bias (Rule 7).
  - Evaluated with RMSE, MAE, and MAPE metrics across all 5 tickers; results saved to `data/processed/baseline_results.csv`.
  - Selected best model overall (Linear Regression with mean Test MAPE of 1.42%) and saved artifacts to `models/best_model.joblib`.
  - Unit tests in `tests/test_features_and_models.py`.
- [ ] **Phase 3: Deep Learning Time-Series Models**
  - Sequential LSTM / GRU architectures for multi-horizon price forecasting.
- [ ] **Phase 4: Financial News Sentiment Analysis (FinBERT)**
  - Financial headlines ingestion and FinBERT sentiment scoring.
- [ ] **Phase 5: Portfolio Optimization & Recommendation Engine**
  - PyPortfolioOpt Modern Portfolio Theory and multi-factor Buy/Hold/Sell decisions.
- [ ] **Phase 6: Interactive Streamlit Dashboard**
  - Plotly interactive charts, model comparison, and portfolio allocation UI.

---

## 📊 Phase 2 Baseline Model Benchmark Results

Evaluated on 5 years of daily data using chronological 80/20 train/test split:

| Ticker | Model | Train RMSE | Test RMSE | Test MAE | Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RELIANCE.NS** | Linear Regression | 17.63 | **20.77** | **14.88** | **1.11%** |
| RELIANCE.NS | Random Forest | 8.09 | 22.50 | 16.94 | 1.25% |
| RELIANCE.NS | XGBoost | 9.25 | 23.36 | 17.73 | 1.32% |
| **TCS.NS** | Linear Regression | 46.80 | **53.06** | **39.27** | **1.60%** |
| TCS.NS | Random Forest | 21.65 | 548.68 | 466.32 | 20.38% |
| TCS.NS | XGBoost | 25.00 | 596.56 | 511.36 | 22.30% |
| **INFY.NS** | Linear Regression | 24.76 | **27.44** | **21.00** | **1.73%** |
| INFY.NS | Random Forest | 11.57 | 129.12 | 98.83 | 8.90% |
| INFY.NS | XGBoost | 13.20 | 129.52 | 100.24 | 9.00% |
| **HDFCBANK.NS** | Linear Regression | 12.48 | 14.01 | 10.49 | 1.32% |
| HDFCBANK.NS | Random Forest | 4.85 | **13.34** | **10.18** | **1.28%** |
| HDFCBANK.NS | XGBoost | 5.65 | 14.39 | 11.37 | 1.43% |
| **ITC.NS** | Linear Regression | 5.42 | **6.24** | **4.04** | **1.35%** |
| ITC.NS | Random Forest | 2.06 | 18.18 | 16.13 | 5.49% |
| ITC.NS | XGBoost | 2.70 | 11.42 | 9.51 | 3.19% |

*Overall best model across all tickers: **Linear Regression** (Mean Test MAPE: **1.42%**), saved to `models/best_model.joblib`.*

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

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```
