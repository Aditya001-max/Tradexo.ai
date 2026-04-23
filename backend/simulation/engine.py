"""
Simulation Engine
=================
Core stateless simulation function.

This is a PURE FUNCTION — no side effects, no shared state.
Designed to be safely executed across multiple processes.

Computes:
- Entry price at shifted entry time
- Exit price at shifted exit time (or stop-loss trigger)
- P&L based on direction and position size
- Maximum drawdown during holding period
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimParams:
    """Parameters for a single counterfactual simulation."""

    entry_shift: int         # minutes to shift entry (negative = earlier)
    exit_shift: int          # minutes to shift exit (negative = earlier)
    stop_loss: float         # fractional stop loss (e.g., 0.01 = 1%)
    size_multiplier: float   # position size multiplier (e.g., 1.5 = 150%)


@dataclass
class SimResult:
    """Result of a single counterfactual simulation."""

    entry_shift: int
    exit_shift: int
    stop_loss: float
    size_multiplier: float
    entry_price: Optional[float]
    exit_price: Optional[float]
    pnl: float
    drawdown: float
    exit_reason: str          # 'time' | 'stop_loss' | 'no_data'


def _get_price_at(
    timestamps: np.ndarray,
    prices: np.ndarray,
    target_ts: float,
) -> Optional[float]:
    """
    Get the price nearest to target timestamp.

    Args:
        timestamps: Array of Unix timestamps
        prices: Array of close prices
        target_ts: Target Unix timestamp

    Returns:
        Nearest price or None
    """
    if len(timestamps) == 0:
        return None

    idx = np.searchsorted(timestamps, target_ts)
    idx = min(idx, len(timestamps) - 1)

    # Check if the previous index is closer
    if idx > 0:
        if abs(timestamps[idx - 1] - target_ts) < abs(timestamps[idx] - target_ts):
            idx = idx - 1

    return float(prices[idx])


def simulate_trade(
    params: dict,
    market_data: dict,
) -> dict:
    """
    Core simulation function — PURE, STATELESS, PROCESS-SAFE.

    Simulates a counterfactual trade with shifted entry/exit,
    different stop-loss, and different position size.

    Args:
        params: Dictionary with keys:
            - entry_shift (int): minutes to shift entry
            - exit_shift (int): minutes to shift exit
            - stop_loss (float): fractional stop loss
            - size_multiplier (float): position size multiplier
            - original_entry_ts (float): original entry Unix timestamp
            - original_exit_ts (float): original exit Unix timestamp
            - direction (str): 'long' or 'short'
            - original_size (float): original position size

        market_data: Dictionary with keys:
            - timestamps (list[float]): Unix timestamps
            - close (list[float]): close prices
            - high (list[float]): high prices
            - low (list[float]): low prices

    Returns:
        Dictionary with simulation results
    """
    # --- Unpack params ---
    entry_shift = params["entry_shift"]
    exit_shift = params["exit_shift"]
    stop_loss = params["stop_loss"]
    size_multiplier = params["size_multiplier"]
    direction = params["direction"]
    original_size = params["original_size"]

    # --- Compute shifted timestamps ---
    shifted_entry_ts = params["original_entry_ts"] + (entry_shift * 60)
    shifted_exit_ts = params["original_exit_ts"] + (exit_shift * 60)

    # Ensure exit is after entry
    if shifted_exit_ts <= shifted_entry_ts:
        return {
            "entry_shift": entry_shift,
            "exit_shift": exit_shift,
            "stop_loss": stop_loss,
            "size_multiplier": size_multiplier,
            "entry_price": None,
            "exit_price": None,
            "pnl": 0.0,
            "drawdown": 0.0,
            "exit_reason": "invalid_timing",
        }

    # --- Convert market data to numpy arrays ---
    timestamps = np.array(market_data["timestamps"])
    close_prices = np.array(market_data["close"])
    high_prices = np.array(market_data["high"])
    low_prices = np.array(market_data["low"])

    # --- Get entry price ---
    entry_price = _get_price_at(timestamps, close_prices, shifted_entry_ts)
    if entry_price is None or entry_price == 0:
        return {
            "entry_shift": entry_shift,
            "exit_shift": exit_shift,
            "stop_loss": stop_loss,
            "size_multiplier": size_multiplier,
            "entry_price": None,
            "exit_price": None,
            "pnl": 0.0,
            "drawdown": 0.0,
            "exit_reason": "no_data",
        }

    # --- Compute stop-loss price ---
    if direction == "long":
        stop_price = entry_price * (1 - stop_loss)
    else:
        stop_price = entry_price * (1 + stop_loss)

    # --- Scan for stop-loss trigger + track drawdown ---
    # Get bars between entry and exit
    entry_mask = timestamps >= shifted_entry_ts
    exit_mask = timestamps <= shifted_exit_ts
    holding_mask = entry_mask & exit_mask

    holding_indices = np.where(holding_mask)[0]

    max_drawdown = 0.0
    stop_triggered = False
    stop_exit_price = None
    running_peak = entry_price

    for idx in holding_indices:
        bar_high = high_prices[idx]
        bar_low = low_prices[idx]
        bar_close = close_prices[idx]

        if direction == "long":
            # Check stop-loss trigger (price went below stop)
            if bar_low <= stop_price:
                stop_triggered = True
                stop_exit_price = stop_price  # assume filled at stop level
                break

            # Track drawdown
            running_peak = max(running_peak, bar_high)
            current_drawdown = (running_peak - bar_low) / running_peak
            max_drawdown = max(max_drawdown, current_drawdown)

        else:  # short
            # Check stop-loss trigger (price went above stop)
            if bar_high >= stop_price:
                stop_triggered = True
                stop_exit_price = stop_price
                break

            # Track drawdown (inverse for short)
            running_peak = min(running_peak, bar_low) if running_peak != entry_price else bar_low
            if running_peak > 0:
                current_drawdown = (bar_high - running_peak) / running_peak
                max_drawdown = max(max_drawdown, abs(current_drawdown))

    # --- Determine exit price ---
    if stop_triggered:
        exit_price = stop_exit_price
        exit_reason = "stop_loss"
    else:
        exit_price = _get_price_at(timestamps, close_prices, shifted_exit_ts)
        exit_reason = "time"

    if exit_price is None:
        exit_price = entry_price  # fallback: no P&L
        exit_reason = "no_data"

    # --- Compute P&L ---
    effective_size = original_size * size_multiplier

    if direction == "long":
        pnl = (exit_price - entry_price) * effective_size
    else:
        pnl = (entry_price - exit_price) * effective_size

    return {
        "entry_shift": entry_shift,
        "exit_shift": exit_shift,
        "stop_loss": stop_loss,
        "size_multiplier": size_multiplier,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "pnl": round(pnl, 2),
        "drawdown": round(max_drawdown, 6),
        "exit_reason": exit_reason,
    }
