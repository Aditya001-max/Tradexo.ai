"""
Ingestion Agent
===============
Agent 1: Validates and normalizes incoming trade data.

Responsibilities:
- Accept trade input (API / JSON)
- Validate data integrity
- Normalize schema to internal format
"""

from datetime import datetime, timezone
from typing import Optional

from backend.schemas import TradeInput
from backend.utils.logger import get_logger

logger = get_logger("ingestion_agent")


# ============================================
# KNOWN ASSETS (expandable)
# ============================================
KNOWN_ASSETS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "V", "MA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "BA",
    "SPY", "QQQ", "IWM", "DIA", "ARKK", "XLF", "XLE", "XLK",
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
    "GC=F", "SI=F", "CL=F",  # Gold, Silver, Oil futures
}


class IngestionAgent:
    """
    Agent 1: Validates and normalizes trade input.

    Takes raw TradeInput from the API and produces a clean,
    validated trade dictionary for downstream agents.
    """

    def __init__(self):
        self.name = "IngestionAgent"

    def validate_and_normalize(self, trade_input: TradeInput) -> dict:
        """
        Validate trade input and return a normalized trade object.

        Args:
            trade_input: Raw trade input from API

        Returns:
            Normalized trade dictionary

        Raises:
            ValueError: If validation fails
        """
        logger.info(
            f"[{self.name}] Validating trade: {trade_input.asset} "
            f"{trade_input.direction} {trade_input.size} units"
        )

        errors = []

        # --- 1. Asset Validation ---
        asset = trade_input.asset.upper().strip()
        if asset not in KNOWN_ASSETS:
            logger.warning(
                f"[{self.name}] Unknown asset '{asset}' — proceeding anyway "
                f"(yfinance may still support it)"
            )

        # --- 2. Direction ---
        direction = trade_input.direction.lower().strip()
        if direction not in ("long", "short"):
            errors.append(f"Invalid direction: '{direction}'. Must be 'long' or 'short'.")

        # --- 3. Timestamps ---
        entry_time = trade_input.entry_time
        exit_time = trade_input.exit_time

        if exit_time <= entry_time:
            errors.append("exit_time must be strictly after entry_time.")

        # Check trade is not in the future
        now = datetime.now(timezone.utc)
        entry_utc = entry_time.replace(tzinfo=timezone.utc) if entry_time.tzinfo is None else entry_time
        if entry_utc > now:
            errors.append("entry_time cannot be in the future.")

        # Check reasonable duration (max 30 days)
        duration = exit_time - entry_time
        if duration.days > 30:
            errors.append(f"Trade duration too long: {duration.days} days. Max is 30 days.")

        # --- 4. Size ---
        if trade_input.size <= 0:
            errors.append("Position size must be positive.")

        if trade_input.size > 1_000_000:
            logger.warning(f"[{self.name}] Very large position size: {trade_input.size}")

        # --- Raise on errors ---
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"[{self.name}] Validation failed: {error_msg}")
            raise ValueError(f"Trade validation failed: {error_msg}")

        # --- 5. Normalize to UTC ---
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        # --- Build clean trade object ---
        clean_trade = {
            "asset": asset,
            "direction": direction,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "size": trade_input.size,
            "duration_minutes": int(duration.total_seconds() / 60),
        }

        logger.info(
            f"[{self.name}] ✅ Trade validated: {asset} {direction} "
            f"| {clean_trade['duration_minutes']}min duration "
            f"| {trade_input.size} units"
        )

        return clean_trade
