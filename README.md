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
  - `EarlyStopping` and `ModelCheckpoint` callbacks to prevent overfitting and capture optimal weights.
  - High-resolution prediction plot saved to `data/processed/lstm_actual_vs_predicted.png`.
  - Comprehensive model comparison table against Phase 2 baselines saved to `data/processed/model_comparison.csv`.
  - Model weights saved to `models/lstm_model.pt` and fitted scaler saved to `models/scaler.pkl`.
  - Unit tests added in `tests/test_lstm_model.py`.
- [ ] **Phase 4: Financial News Sentiment Analysis (FinBERT)**
  - Financial headlines ingestion and FinBERT sentiment scoring.
- [ ] **Phase 5: Portfolio Optimization & Recommendation Engine**
  - PyPortfolioOpt Modern Portfolio Theory and multi-factor Buy/Hold/Sell decisions.
- [ ] **Phase 6: Interactive Streamlit Dashboard**
  - Plotly interactive charts, model comparison, and portfolio allocation UI.

---

## 📊 Model Comparison Benchmark (RELIANCE.NS)

Evaluated on 5 years of historical data using chronological 80/20 train/test split:

| Model Type | Architecture / Estimator | Test RMSE | Test MAE | Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline ML** | Linear Regression | 20.77 | 14.88 | 1.11% |
| **Baseline ML** | Random Forest Regressor | 22.50 | 16.94 | 1.25% |
| **Baseline ML** | XGBoost Regressor | 23.36 | 17.73 | 1.32% |
| **Deep Learning** | **LSTM (60-Day Sequence Window)** | **30.42** | **23.68** | **1.75%** |

*Artifacts: PyTorch model saved to `models/lstm_model.pt`, scaler saved to `models/scaler.pkl`, and comparison chart saved to `data/processed/lstm_actual_vs_predicted.png`.*

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

### 4. Run All Unit Tests
```bash
python -m unittest discover tests
```
