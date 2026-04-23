"""
Aggregation Agent
=================
Agent 5: Combines simulation results into structured outputs.

Responsibilities:
- Build P&L heatmap matrices (entry_shift × exit_shift)
- Identify best trade and top 3 alternatives
- Compute summary metrics (best/worst/median P&L, drawdowns)
- Produce visualization-ready data structures
"""

from typing import Any

import numpy as np

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("aggregation_agent")
settings = get_settings()


class AggregationAgent:
    """
    Agent 5: Results aggregator.

    Takes raw simulation results and produces:
    1. P&L heatmap matrices for each (stop_loss, size_multiplier) combo
    2. Best overall trade
    3. Top 3 alternative trades
    4. Summary metrics
    """

    def __init__(self):
        self.name = "AggregationAgent"

    def aggregate(
        self,
        results: list[dict],
        actual_pnl: float = 0.0,
    ) -> dict:
        """
        Aggregate all simulation results.

        Args:
            results: List of simulation result dictionaries
            actual_pnl: P&L of the actual trade for comparison

        Returns:
            Aggregated output dictionary with heatmaps, rankings, and metrics
        """
        logger.info(f"[{self.name}] Aggregating {len(results)} simulation results...")

        if not results:
            logger.error(f"[{self.name}] No results to aggregate")
            return self._empty_output()

        # --- Filter valid results ---
        valid = [r for r in results if r.get("exit_reason") not in ("no_data", "error", "invalid_timing")]
        logger.info(f"[{self.name}] {len(valid)} valid results out of {len(results)}")

        if not valid:
            return self._empty_output()

        # --- Sort by P&L descending ---
        sorted_results = sorted(valid, key=lambda r: r["pnl"], reverse=True)

        # --- Best trade ---
        best = sorted_results[0]
        best_trade = {
            **best,
            "improvement": self._compute_improvement(actual_pnl, best["pnl"]),
        }

        # --- Top 3 ---
        top_3 = []
        for r in sorted_results[:3]:
            top_3.append({
                **r,
                "improvement": self._compute_improvement(actual_pnl, r["pnl"]),
            })

        # --- Heatmaps ---
        heatmaps = self._build_heatmaps(valid)

        # --- Metrics ---
        all_pnls = [r["pnl"] for r in valid]
        all_drawdowns = [r["drawdown"] for r in valid]

        metrics = {
            "total_simulations": len(results),
            "valid_simulations": len(valid),
            "best_pnl": round(max(all_pnls), 2),
            "worst_pnl": round(min(all_pnls), 2),
            "median_pnl": round(float(np.median(all_pnls)), 2),
            "mean_pnl": round(float(np.mean(all_pnls)), 2),
            "std_pnl": round(float(np.std(all_pnls)), 2),
            "actual_pnl": round(actual_pnl, 2),
            "max_drawdown": round(max(all_drawdowns), 6) if all_drawdowns else 0.0,
            "avg_drawdown": round(float(np.mean(all_drawdowns)), 6) if all_drawdowns else 0.0,
            "profitable_pct": round(
                len([p for p in all_pnls if p > 0]) / len(all_pnls) * 100, 1
            ),
            "improvement_over_actual": self._compute_improvement(actual_pnl, max(all_pnls)),
        }

        logger.info(
            f"[{self.name}] ✅ Aggregation complete: "
            f"best P&L={metrics['best_pnl']}, "
            f"median={metrics['median_pnl']}, "
            f"profitable={metrics['profitable_pct']}%"
        )

        return {
            "best_trade": best_trade,
            "top_3": top_3,
            "heatmaps": heatmaps,
            "metrics": metrics,
            "all_results": valid,
        }

    def _build_heatmaps(self, results: list[dict]) -> list[dict]:
        """
        Build P&L heatmap matrices.

        Creates one heatmap per (stop_loss, size_multiplier) pair.
        Each heatmap is entry_shift × exit_shift → average P&L.

        Args:
            results: Valid simulation results

        Returns:
            List of heatmap data dictionaries
        """
        entry_shifts = sorted(set(r["entry_shift"] for r in results))
        exit_shifts = sorted(set(r["exit_shift"] for r in results))

        # --- Group by (stop_loss, size_multiplier) ---
        groups: dict[tuple, list] = {}
        for r in results:
            key = (r["stop_loss"], r["size_multiplier"])
            groups.setdefault(key, []).append(r)

        heatmaps = []

        # Build a heatmap using the default parameters (original size, middle stop-loss)
        # Also build one for each unique combination
        default_sl = settings.STOP_LOSS_OPTIONS[len(settings.STOP_LOSS_OPTIONS) // 2]  # middle
        default_sm = 1.0  # original size

        # Prioritize: show default combo first, then others
        sorted_keys = sorted(groups.keys(), key=lambda k: (k != (default_sl, default_sm), k))

        for stop_loss, size_mult in sorted_keys:
            group = groups[(stop_loss, size_mult)]

            # Build lookup: (entry_shift, exit_shift) → P&L
            pnl_lookup: dict[tuple, list] = {}
            for r in group:
                key = (r["entry_shift"], r["exit_shift"])
                pnl_lookup.setdefault(key, []).append(r["pnl"])

            # Build matrix
            matrix = []
            for es in entry_shifts:
                row = []
                for xs in exit_shifts:
                    pnls = pnl_lookup.get((es, xs), [0.0])
                    row.append(round(float(np.mean(pnls)), 2))
                matrix.append(row)

            heatmaps.append({
                "entry_shifts": entry_shifts,
                "exit_shifts": exit_shifts,
                "pnl_matrix": matrix,
                "stop_loss": stop_loss,
                "size_multiplier": size_mult,
            })

        logger.info(f"[{self.name}] Built {len(heatmaps)} heatmap matrices")
        return heatmaps

    def _compute_improvement(self, actual_pnl: float, best_pnl: float) -> str:
        """Compute improvement percentage string."""
        if actual_pnl == 0:
            if best_pnl > 0:
                return f"+${best_pnl:.2f}"
            elif best_pnl < 0:
                return f"-${abs(best_pnl):.2f}"
            return "0%"

        pct = ((best_pnl - actual_pnl) / abs(actual_pnl)) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    def _empty_output(self) -> dict:
        """Return empty aggregated output."""
        return {
            "best_trade": None,
            "top_3": [],
            "heatmaps": [],
            "metrics": {
                "total_simulations": 0,
                "valid_simulations": 0,
                "best_pnl": 0.0,
                "worst_pnl": 0.0,
                "median_pnl": 0.0,
                "mean_pnl": 0.0,
                "std_pnl": 0.0,
                "actual_pnl": 0.0,
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "profitable_pct": 0.0,
                "improvement_over_actual": "N/A",
            },
            "all_results": [],
        }
