"""Application configuration and constants."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Project directory structure
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
MODELS_DIR: Path = ROOT_DIR / "models"

# Ensure runtime directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default Market Data Parameters
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
DEFAULT_PERIOD: str = "3y"
DEFAULT_INTERVAL: str = "1d"

# Chronological Time-Series Split Parameters (Never Random)
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# API Configuration
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
