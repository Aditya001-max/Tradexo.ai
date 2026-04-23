"""
Structured Logger
=================
Centralized logging with JSON formatting and per-agent context.
"""

import logging
import sys
from pythonjsonlogger import jsonlogger

from backend.config import get_settings

settings = get_settings()

# --- Formatter ---
_formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# --- Stream Handler ---
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Get a structured logger for a specific agent/module.

    Args:
        name: Logger name (e.g., 'ingestion_agent', 'simulation_engine')

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"ctae.{name}")
    if not logger.handlers:
        logger.addHandler(_handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
