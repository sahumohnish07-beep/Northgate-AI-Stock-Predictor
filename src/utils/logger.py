"""Centralized logging module for Northgate AI Stock Predictor."""

import logging
import sys
from typing import Optional


def get_logger(
    name: str = "northgate_ai", level: Optional[int] = None
) -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name: Name of the logger.
        level: Optional custom logging level (defaults to INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level or logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
