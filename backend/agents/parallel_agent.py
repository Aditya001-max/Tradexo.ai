"""
Parallel Execution Agent (Antigravity Controller)
===================================================
Agent 4: Distributes simulation tasks across processes.

Responsibilities:
- Distribute simulation parameter sets to workers
- Execute using map/reduce pattern with ProcessPoolExecutor
- Collect and return all results
- Ensure stateless, process-safe execution

This is the "Antigravity" parallel execution layer.
Uses concurrent.futures.ProcessPoolExecutor as the backend.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from backend.simulation.engine import simulate_trade
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("parallel_agent")
settings = get_settings()


def _worker_simulate(args: tuple) -> dict:
    """
    Top-level worker function for ProcessPoolExecutor.

    Must be defined at module level (not as a method) for pickling.

    Args:
        args: Tuple of (params_dict, market_data_dict)

    Returns:
        Simulation result dictionary
    """
    params, market_data = args
    return simulate_trade(params, market_data)


class ParallelAgent:
    """
    Agent 4: Parallel execution controller.

    Distributes simulation jobs across multiple CPU cores using
    ProcessPoolExecutor (map/reduce pattern).

    This replaces Celery + Redis for local execution while maintaining
    the same interface — swap in Celery for distributed deployment.
    """

    def __init__(self, max_workers: int = None):
        self.name = "ParallelAgent"
        self.max_workers = max_workers or settings.MAX_WORKERS

    def _prepare_market_data(self, df: pd.DataFrame) -> dict:
        """
        Convert DataFrame to a serializable dictionary for cross-process transfer.

        ProcessPoolExecutor uses pickle — DataFrames are serializable but
        plain dicts of numpy arrays are faster to transfer.

        Args:
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            Dict with lists of floats (timestamps, OHLCV)
        """
        return {
            "timestamps": [ts.timestamp() for ts in df.index],
            "open": df["Open"].tolist(),
            "high": df["High"].tolist(),
            "low": df["Low"].tolist(),
            "close": df["Close"].tolist(),
            "volume": df["Volume"].tolist() if "Volume" in df.columns else [],
        }

    def run_counterfactuals(
        self,
        param_grid: list[dict],
        market_data_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Execute all counterfactual simulations in parallel.

        This is the core map/reduce operation:
        1. MAP: Distribute simulate_trade(params, market_data) across workers
        2. REDUCE: Collect all results into a list

        Args:
            param_grid: List of parameter dictionaries from SimulationAgent
            market_data_df: OHLCV DataFrame from MarketDataAgent

        Returns:
            List of simulation result dictionaries
        """
        total = len(param_grid)
        logger.info(
            f"[{self.name}] ⚡ Starting parallel execution: "
            f"{total} simulations across {self.max_workers} workers"
        )

        # --- Serialize market data once (shared across all workers) ---
        market_data = self._prepare_market_data(market_data_df)

        # --- Prepare job arguments ---
        job_args = [(params, market_data) for params in param_grid]

        # --- Execute with ProcessPoolExecutor ---
        start_time = time.perf_counter()
        results = []
        failed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # MAP: submit all jobs
            future_to_idx = {
                executor.submit(_worker_simulate, args): idx
                for idx, args in enumerate(job_args)
            }

            # REDUCE: collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"[{self.name}] Simulation {idx} failed: {e}"
                    )
                    # Append a failure marker
                    results.append({
                        "entry_shift": param_grid[idx].get("entry_shift", 0),
                        "exit_shift": param_grid[idx].get("exit_shift", 0),
                        "stop_loss": param_grid[idx].get("stop_loss", 0),
                        "size_multiplier": param_grid[idx].get("size_multiplier", 1),
                        "entry_price": None,
                        "exit_price": None,
                        "pnl": 0.0,
                        "drawdown": 0.0,
                        "exit_reason": "error",
                    })

        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        logger.info(
            f"[{self.name}] ✅ Completed {len(results)} simulations in "
            f"{elapsed:.1f}ms ({failed} failures)"
        )

        return results, elapsed

    def run_sequential_fallback(
        self,
        param_grid: list[dict],
        market_data_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Sequential fallback for debugging or when multiprocessing is unavailable.

        Args:
            param_grid: List of parameter dictionaries
            market_data_df: OHLCV DataFrame

        Returns:
            List of simulation result dictionaries
        """
        logger.warning(f"[{self.name}] Running in SEQUENTIAL mode (fallback)")

        market_data = self._prepare_market_data(market_data_df)
        start_time = time.perf_counter()

        results = []
        for params in param_grid:
            try:
                result = simulate_trade(params, market_data)
                results.append(result)
            except Exception as e:
                logger.error(f"[{self.name}] Simulation failed: {e}")
                results.append({
                    "entry_shift": params.get("entry_shift", 0),
                    "exit_shift": params.get("exit_shift", 0),
                    "stop_loss": params.get("stop_loss", 0),
                    "size_multiplier": params.get("size_multiplier", 1),
                    "entry_price": None,
                    "exit_price": None,
                    "pnl": 0.0,
                    "drawdown": 0.0,
                    "exit_reason": "error",
                })

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{self.name}] Sequential execution: {len(results)} sims in {elapsed:.1f}ms")

        return results, elapsed
