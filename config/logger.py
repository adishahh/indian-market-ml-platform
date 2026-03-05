"""
Centralized logging configuration for Indian Market ML Platform.
Usage in any module:
    from config.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Daily rotating log file
LOG_FILE = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with both file and console handlers.
    File logs go to logs/pipeline_YYYYMMDD.log
    Console logs show WARNING and above by default.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured, avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    # --- File Handler (DEBUG and above) ---
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    # --- Console Handler (INFO and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
