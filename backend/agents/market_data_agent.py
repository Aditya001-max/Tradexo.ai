"""
Market Data Agent
=================
Agent 2: Fetches OHLCV market data from Yahoo Finance.

Responsibilities:
- Fetch historical OHLCV data around the trade window
- Auto-select interval based on trade age
- Cache results to avoid redundant API calls
- Provide a time buffer for counterfactual simulations
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from backend.utils.cache import cache_get, cache_set
from backend.utils.logger import get_logger

logger = get_logger("market_data_agent")


class MarketDataAgent:
    """
    Agent 2: Fetches and caches OHLCV market data.

    Automatically selects the best interval based on how old the trade is,
    and adds buffer time before/after the trade for counterfactual analysis.
    """

    # Interval limits imposed by Yahoo Finance
    INTERVAL_LIMITS = {
        "1m": timedelta(days=7),
        "5m": timedelta(days=60),
        "15m": timedelta(days=60),
        "1h": timedelta(days=730),
        "1d": timedelta(days=3650),
    }

    def __init__(self):
        self.name = "MarketDataAgent"

    def _select_interval(self, entry_time: datetime) -> str:
        """
        Select the finest available interval based on trade age.

        Args:
            entry_time: Trade entry timestamp

        Returns:
            Interval string (e.g., '1m', '5m', '1h')
        """
        now = datetime.now(timezone.utc)
        entry_utc = entry_time.replace(tzinfo=timezone.utc) if entry_time.tzinfo is None else entry_time
        age = now - entry_utc

        for interval, max_age in self.INTERVAL_LIMITS.items():
            if age <= max_age:
                logger.info(f"[{self.name}] Selected interval: {interval} (trade age: {age.days}d)")
                return interval

        logger.warning(f"[{self.name}] Trade is very old ({age.days} days), using daily interval")
        return "1d"

    def _compute_buffer(self, entry_time: datetime, exit_time: datetime) -> tuple[datetime, datetime]:
        """
        Compute the data fetch window with buffer for counterfactual shifts.

        We add 2 hours before entry and 2 hours after exit to allow
        for ±60 min entry/exit shifts plus some safety margin.

        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time

        Returns:
            (start_time, end_time) with buffer
        """
        buffer = timedelta(hours=2)
        start = entry_time - buffer
        end = exit_time + buffer
        return start, end

    def fetch(
        self,
        asset: str,
        entry_time: datetime,
        exit_time: datetime,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given asset and time window.

        Args:
            asset: Ticker symbol (e.g., 'AAPL')
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            interval: Override interval (auto-detected if None)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex (UTC)

        Raises:
            ValueError: If no data could be fetched
        """
        # --- Select interval ---
        if interval is None:
            interval = self._select_interval(entry_time)

        # --- Compute window ---
        start, end = self._compute_buffer(entry_time, exit_time)

        # --- Check cache ---
        cache_key = f"market_{asset}_{start.isoformat()}_{end.isoformat()}_{interval}"
        cached = cache_get(cache_key)
        if cached is not None:
            logger.info(f"[{self.name}] Cache hit for {asset} ({interval})")
            return cached

        # --- Fetch from Yahoo Finance ---
        logger.info(
            f"[{self.name}] Fetching {asset} OHLCV data: "
            f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} "
            f"@ {interval} interval"
        )

        try:
            ticker = yf.Ticker(asset)
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"[{self.name}] yfinance fetch failed: {e}")
            raise ValueError(f"Failed to fetch market data for {asset}: {e}")

        # --- Validate ---
        if df is None or df.empty:
            error_msg = (
                f"No market data returned for {asset} between "
                f"{start.isoformat()} and {end.isoformat()} at {interval} interval. "
                f"This may be outside market hours or the asset may not exist."
            )
            logger.error(f"[{self.name}] {error_msg}")
            raise ValueError(error_msg)

        # --- Clean ---
        # Ensure timezone-aware UTC index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Keep only needed columns
        columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in columns_to_keep if c in df.columns]
        df = df[available]

        # Drop rows with NaN prices
        df = df.dropna(subset=["Close"])

        logger.info(
            f"[{self.name}] ✅ Fetched {len(df)} bars for {asset} "
            f"({df.index[0]} → {df.index[-1]})"
        )

        # --- Cache ---
        cache_set(cache_key, df)

        return df

    def get_price_at_time(
        self,
        df: pd.DataFrame,
        target_time: datetime,
        method: str = "nearest",
    ) -> Optional[float]:
        """
        Get the closing price at or nearest to a specific time.

        Args:
            df: OHLCV DataFrame
            target_time: Target timestamp
            method: Lookup method ('nearest', 'ffill', 'bfill')

        Returns:
            Price at the target time, or None if unavailable
        """
        if df.empty:
            return None

        target = pd.Timestamp(target_time, tz="UTC")

        if method == "nearest":
            idx = df.index.get_indexer([target], method="nearest")[0]
            if idx < 0 or idx >= len(df):
                return None
            return float(df.iloc[idx]["Close"])
        elif method == "ffill":
            idx = df.index.get_indexer([target], method="ffill")[0]
            if idx < 0:
                return None
            return float(df.iloc[idx]["Close"])
        elif method == "bfill":
            idx = df.index.get_indexer([target], method="bfill")[0]
            if idx < 0 or idx >= len(df):
                return None
            return float(df.iloc[idx]["Close"])

        return None
