"""
Configuration & Settings
========================
Central configuration using pydantic-settings.
Reads from environment variables and .env file.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # --- Project ---
    PROJECT_NAME: str = "Counterfactual Trade Analysis Engine"
    VERSION: str = "1.0.0"

    # --- Hugging Face ---
    HF_API_TOKEN: str = ""

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./trades.db"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Parallel Processing ---
    MAX_WORKERS: int = os.cpu_count() or 4

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Simulation Defaults ---
    ENTRY_SHIFT_RANGE: list[int] = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    EXIT_SHIFT_RANGE: list[int] = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    STOP_LOSS_OPTIONS: list[float] = [0.0025, 0.005, 0.01, 0.02]
    SIZE_MULTIPLIERS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5]

    # --- Cache ---
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 256

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
