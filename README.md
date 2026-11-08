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
- [x] **Phase 5: Portfolio Optimization & Recommendation Engine**
  - Built `src/portfolio.py` using Modern Portfolio Theory (MPT) via `PyPortfolioOpt`.
  - Computed annualized expected returns ($\mu$) and covariance matrix ($\Sigma$) across 5 premier Indian equities.
  - Implemented 3 investor risk profiles:
    - **Low Risk**: Minimum Volatility portfolio ($w_i \ge 0$).
    - **Medium Risk**: Maximum Sharpe Ratio tangency portfolio ($R_f = 6.0\%$).
    - **High Risk**: Maximum Return subject to a single-asset weight cap ($w_i \le 40\%$).
  - Exported allocation metrics to `data/processed/portfolio_allocations.csv`.
  - Generated and saved Efficient Frontier chart with risk profile markers to `data/processed/efficient_frontier.png`.
  - Added multi-factor algorithmic Buy / Hold / Sell / Underweight recommendation framework.
  - Comprehensive unit test suite in `tests/test_portfolio.py`.
- [x] **Phase 6: Recommendation Engine & Backtesting**
  - Built `src/recommender.py` fusing 5-day predicted return, FinBERT sentiment, and volatility.
  - Rule-based decision logic: Buy if predicted return > +2% and sentiment > 0.1; Sell if < -2% and sentiment < -0.1; otherwise Hold.
  - Risk-profile adaptive threshold customization across Low (conservative), Medium (balanced), and High (aggressive) profiles.
  - Granular confidence level estimation (0.0 to 1.0) and descriptive diagnostic rationale.
  - Walk-forward, lookahead-free backtesting framework comparing strategy returns against buy-and-hold benchmarks.
  - Evaluation metrics: Total Return, CAGR, Annualized Volatility, Sharpe Ratio, and Maximum Drawdown.
  - Exported artifacts: `data/processed/backtest_results.csv` and dual-panel chart `data/processed/backtest_results.png`.
  - Comprehensive unit test suite in `tests/test_recommender.py`.
- [x] **Phase 7: Interactive Streamlit Dashboard**
  - Full-featured multi-tab enterprise dashboard in `app.py`.
  - Sidebar: stock selector (5 premier Indian equities), risk profile selector (Low, Medium, High), forecast horizon slider (1–15 days), and forecasting model engine toggle.
  - 5 interactive tabs:
    - **📊 Market Overview**: Real-time KPI tiles, interactive OHLC candlestick chart with SMA (20, 50), EMA (20), and Bollinger Bands, volume subplots, RSI (14) gauge, and recent OHLCV records table.
    - **🔮 Price Prediction**: Multi-step autoregressive price projections, 95% confidence interval shaded cones, and model test benchmarks (MAE, RMSE, MAPE).
    - **📰 News Sentiment (FinBERT)**: Real-time polarity score meter, historical daily sentiment trend bar chart, and curated financial news table.
    - **💼 Portfolio Optimization**: Modern Portfolio Theory (PyPortfolioOpt), interactive allocation donut chart, dynamic Efficient Frontier curve, and cross-profile weight matrix.
    - **🎯 Recommendations & Backtest**: Prominent styled recommendation card (🟢 Buy, 🟡 Hold, 🔴 Sell) with confidence meter, rationale string, threshold pills, and walk-forward lookahead-free backtester with equity curves and underwater drawdowns.
  - Performance caching via `@st.cache_data` and `@st.cache_resource`, loading spinners, and robust error handling.
  - Prominent educational and risk disclaimer in footer.

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

## 📊 Phase 5 Portfolio Optimization Benchmark & Risk Profiles

Portfolio optimized across the 5 premier Indian equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ITC.NS`) over 5-year historical trading data with risk-free rate $R_f = 6.0\%$:

| Risk Profile | Objective | Expected Annual Return | Annual Volatility | Sharpe Ratio | Top Asset Allocations |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Low Risk** | Min Volatility | 3.17% | 14.32% | -0.20 | ITC (35.5%), HDFCBANK (26.0%), TCS (22.0%), RELIANCE (14.3%) |
| **Medium Risk** | Max Sharpe Ratio | 9.56% | 19.76% | +0.18 | ITC (100.0%) |
| **High Risk** | Max Return (40% Cap) | 5.30% | 15.95% | -0.04 | ITC (40.0%), RELIANCE (40.0%), HDFCBANK (20.0%) |

### Algorithmic Allocation & Stock Recommendations

| Ticker | Mean Weight | Low Risk | Medium Risk | High Risk | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ITC.NS** | 58.50% | 35.51% | 100.00% | 40.00% | **Strong Buy (Overweight)** |
| **RELIANCE.NS** | 18.09% | 14.27% | 0.00% | 40.00% | **Buy / Moderate Allocation** |
| **HDFCBANK.NS** | 15.34% | 26.02% | 0.00% | 20.00% | **Buy / Moderate Allocation** |
| **TCS.NS** | 7.34% | 22.02% | 0.00% | 0.00% | **Hold / Selective** |
| **INFY.NS** | 0.72% | 2.17% | 0.00% | 0.00% | **Underweight / Avoid** |

*Data & Visualization Artifacts:*
- Allocations dataset: `data/processed/portfolio_allocations.csv`
- Efficient Frontier plot: `data/processed/efficient_frontier.png`

---

## 📈 Phase 6 Recommendation Engine & Backtesting Benchmark

Evaluated on **RELIANCE.NS** (1,237 trading days) using walk-forward, lookahead-free strategy simulation with benchmark risk-free rate $R_f = 6.0\%$:

| Performance Metric | Recommendation Strategy (Medium Risk) | Buy & Hold Benchmark | Outperformance / Alpha |
| :--- | :---: | :---: | :---: |
| **Total Return (%)** | **+33.16%** | +13.60% | **+19.56%** |
| **CAGR (%)** | **6.04%** | 2.64% | **+3.40%** |
| **Annualized Volatility (%)** | **1.05%** | 22.93% | **-21.88%** |
| **Sharpe Ratio** | **-0.13** | -0.03 | -0.09 |
| **Maximum Drawdown (%)** | **-2.03%** | -27.67% | **+25.64% (Risk Curbed)** |

*Key Findings:*
- **Capital Preservation**: By transitioning to cash ($R_f$) during prolonged downtrends and sideways chop, maximum drawdown was reduced from **-27.67% down to -2.03%**.
- **Net Alpha**: The strategy delivered a **+19.56% excess cumulative return** over the passive buy-and-hold benchmark.

*Data & Visualization Artifacts:*
- Metrics comparison table: `data/processed/backtest_results.csv`
- Dual-panel equity curve & underwater drawdown plot: `data/processed/backtest_results.png`

---

## 🖥️ Phase 7 Interactive Streamlit Dashboard

Launch the unified intelligence interface locally:

```bash
streamlit run app.py
```

### Dashboard Capabilities & Views

1. **Market Overview**:
   - Live KPI metric cards: Current price, absolute & percentage price delta, 52-week trading range, 20-day annualized volatility, and trading volume.
   - Interactive Plotly OHLC candlestick chart with selectable timeframes (3M, 6M, 1Y, 5Y).
   - Moving average overlays (SMA 20, SMA 50, EMA 20) and Bollinger Bands with shaded volatility envelopes.
   - Volume subplots color-coded by bullish/bearish trading sessions.
   - Technical momentum gauges (RSI 14 with overbought/oversold boundaries) and recent daily OHLCV tabular view.

2. **Price Prediction**:
   - Multi-step price trajectory forecasts (1, 3, 5, 10, or 15 days ahead).
   - Dual forecasting engines: Best Machine Learning model (Linear Regression) and Deep Learning recurrent model (StockLSTM).
   - Dynamic 95% confidence intervals expanding over the forecast horizon.
   - Historical model evaluation benchmarks (MAE, RMSE, MAPE).

3. **News Sentiment (FinBERT)**:
   - Polarity classification badge (🟢 Bullish, 🟡 Neutral, 🔴 Bearish) with raw FinBERT polarity score $[-1.0, +1.0]$.
   - Daily historical sentiment trend bar chart with threshold guide lines.
   - Recent curated financial headlines with source attribution and model confidence scores.

4. **Portfolio Optimization**:
   - Modern Portfolio Theory (PyPortfolioOpt) across the 5 default Indian equities.
   - Interactive donut chart showing asset allocation weights tailored to the active investor risk profile.
   - Interactive Efficient Frontier curve plotting individual stock risk-returns, profile optima, and the risk-free rate ($R_f = 6.0\%$).
   - Cross-profile allocation matrix and algorithmic stock recommendations.

5. **Recommendations & Backtest**:
   - Large styled recommendation banner with action badge (🟢 Buy, 🟡 Hold, 🔴 Sell), confidence meter, and comprehensive diagnostic rationale.
   - Walk-forward lookahead-free backtester with configurable capital and parameters.
   - Comparative performance metrics vs Buy-and-Hold benchmark.
   - Dual-panel Plotly interactive chart tracking cumulative portfolio valuations and peak-to-trough drawdowns.

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

### 5. Run Portfolio Optimization
```bash
python src/portfolio.py
```

### 6. Run Recommendation Engine & Backtest
```bash
python src/recommender.py
```

### 7. Launch Interactive Dashboard
```bash
streamlit run app.py
```

### 8. Run All Unit Tests
```bash
python -m unittest discover tests
```
