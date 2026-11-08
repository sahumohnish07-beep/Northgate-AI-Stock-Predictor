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
- [ ] **Phase 2: Technical Indicators & Feature Engineering**
  - RSI, MACD, Bollinger Bands, Moving Averages, and ATR.
  - Chronological time-series split matrices (Rule 7: Never random, strictly temporal).
- [ ] **Phase 3: Machine Learning Forecasting Models**
  - XGBoost, Random Forest, and baseline regressors with walk-forward CV.
- [ ] **Phase 4: Deep Learning Time-Series Models**
  - Sequential LSTM / GRU architectures for multi-horizon price forecasting.
- [ ] **Phase 5: Financial News Sentiment Analysis (FinBERT)**
  - Financial headlines ingestion and FinBERT sentiment scoring.
- [ ] **Phase 6: Portfolio Optimization & Recommendation Engine**
  - PyPortfolioOpt Modern Portfolio Theory and multi-factor Buy/Hold/Sell decisions.
- [ ] **Phase 7: Interactive Streamlit Dashboard**
  - Plotly interactive charts, model comparison, and portfolio allocation UI.

---

## 📁 Project Structure

```text
Northgate-AI-Stock-Predictor/
├── app.py                     # Main Streamlit web application
├── data/
│   ├── raw/                   # 5-year OHLCV CSVs (e.g., RELIANCE.NS.csv) [git-ignored]
│   └── processed/             # Feature-engineered datasets
├── models/                    # Serialized model checkpoints (*.pkl, *.h5)
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis & visual charts
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration constants & paths
│   ├── data_loader.py         # Multi-ticker yfinance downloader & CSV loader
│   ├── data/                  # Ingestion, caching, and time-series splitter
│   └── utils/                 # Standardized logging
├── tests/
│   ├── test_data_loader.py    # Unit tests for data loading & CSV persistence
│   └── test_data_pipeline.py  # Time-series temporal split integrity tests
├── .env.example               # Template environment configuration
├── .gitignore                 # Ignore rules for credentials, data, and cache
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Quick Start

### 1. Installation
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

### 2. Download 5-Year Historical Market Data
To fetch 5 years of daily OHLCV data for the default Indian stocks (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ITC.NS`):
```bash
python src/data_loader.py
```

### 3. Run Unit Tests
```bash
python -m unittest tests/test_data_loader.py
```

### 4. Explore the EDA Notebook
Launch Jupyter to explore `notebooks/01_eda.ipynb`:
```bash
jupyter notebook notebooks/01_eda.ipynb
```
