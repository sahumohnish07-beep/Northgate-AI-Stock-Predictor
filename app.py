"""Northgate AI Stock Predictor - Interactive Dashboard Application.

Educational purpose only, not financial advice.
"""

import streamlit as st

st.set_page_config(
    page_title="Northgate AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Northgate AI Stock Predictor")
st.caption("AI-Powered Stock Forecasting, Sentiment Analysis & Portfolio Optimization")

st.markdown(
    """
    ### Welcome to Northgate AI Stock Predictor
    
    This platform delivers data-driven stock intelligence:
    - **Machine Learning & Deep Learning Forecasting**: Predictive models for stock price movements.
    - **FinBERT News Sentiment**: Real-time polarity scoring from financial headlines.
    - **Modern Portfolio Theory (MPT)**: Sentiment-tilted Sharpe ratio optimization.
    - **Actionable Decision Intelligence**: Multi-factor Buy / Hold / Sell recommendations.
    
    ---
    > ⚠️ **Disclaimer:** *This software is developed strictly for educational and research purposes only. It does not constitute investment or financial advice.*
    """
)
