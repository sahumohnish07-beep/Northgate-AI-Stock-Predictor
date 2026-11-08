# Northgate AI Stock Predictor

An enterprise-grade, end-to-end AI platform that predicts stock prices using Machine Learning and Deep Learning, evaluates financial news sentiment using FinBERT, optimizes investment portfolios with Modern Portfolio Theory, and delivers algorithmic Buy/Hold/Sell recommendations through an interactive Streamlit dashboard.

---

> ⚠️ **IMPORTANT DISCLAIMER**  
> **This platform is for educational and research purposes only, and does not constitute financial or investment advice.** Always conduct your own due diligence before executing any trades or investments.

---

## 🚀 Planned Features

1. **Historical Data Ingestion & Time-Series Pipeline**
   - High-throughput OHLCV market data fetching via `yfinance`.
   - Strict chronological time-series splitting (70% train, 15% validation, 15% test) to prevent lookahead bias and temporal leakage.
   - Parquet disk caching for offline and fast re-computation.

2. **Technical Feature Engineering**
   - Momentum, trend, volatility, and volume indicators (RSI, MACD, Bollinger Bands, Moving Averages, ATR).
   - Time-lagged feature matrices and stationary transformations.

3. **Machine Learning & Deep Learning Forecasting (ML + DL)**
   - Regression and classification forecasting models (Linear Models, Random Forest, XGBoost).
   - Sequential deep learning architectures (LSTM / GRU) for multi-horizon price predictions.
   - Expanding and rolling walk-forward time-series cross-validation.

4. **Financial News Sentiment Analysis (FinBERT)**
   - Ingestion of financial news headlines for target tickers.
   - Pre-trained transformer sentiment scoring (`ProsusAI/finbert`) yielding positive, negative, and neutral confidence scores.

5. **Portfolio Optimization & Recommendation Engine**
   - Modern Portfolio Theory (MPT) and Efficient Frontier optimization using `PyPortfolioOpt`.
   - Sentiment-tilted Sharpe ratio maximization and risk-adjusted asset allocation.
   - Algorithmic Buy / Hold / Sell decision engine combining model predictions, technical signals, and sentiment.

6. **Interactive Streamlit Web Application**
   - Interactive candlestick and indicator charts powered by Plotly.
   - Model comparison dashboards and sentiment breakdown gauges.
   - Portfolio allocation simulator and trading recommendation cards.

---

## 📁 Project Structure

```text
Northgate-AI-Stock-Predictor/
├── app.py                  # Main Streamlit web application
├── data/
│   ├── raw/                # Untouched raw data downloads (ignored by git)
│   └── processed/          # Cleaned, feature-engineered datasets
├── models/                 # Serialized model weights and checkpoints (*.pkl, *.h5)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── config.py           # Configuration parameters and constants
│   ├── data/               # Ingestion, caching, and time-series splitters
│   └── utils/              # Logging and utility helpers
├── tests/                  # Unit and integration test suites
├── .env.example            # Sample environment variables
├── .gitignore              # Files and directories excluded from git
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/sahumohnish07-beep/Northgate-AI-Stock-Predictor.git
cd Northgate-AI-Stock-Predictor
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Run Streamlit Dashboard
```bash
streamlit run app.py
```
