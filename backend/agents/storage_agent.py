"""
Storage Agent
=============
Agent 9: Handles all database persistence operations.

Responsibilities:
- Save trades, simulation results, patterns, and analysis outputs
- Retrieve results by trade_id
- Handle serialization of complex objects
"""

import json
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import async_session_factory
from backend.models import Trade, SimulationResult, Pattern, AnalysisOutput
from backend.utils.logger import get_logger

logger = get_logger("storage_agent")


class StorageAgent:
    """
    Agent 9: Database persistence layer.

    Handles all CRUD operations for trades, results, patterns,
    and aggregated analysis outputs.
    """

    def __init__(self):
        self.name = "StorageAgent"

    async def update_trade_status(
        self,
        trade_id: str,
        status: str,
        error_message: Optional[str] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        actual_pnl: Optional[float] = None,
    ):
        """Update trade status and optional fields."""
        async with async_session_factory() as session:
            values = {"status": status}
            if error_message is not None:
                values["error_message"] = error_message
            if entry_price is not None:
                values["entry_price"] = entry_price
            if exit_price is not None:
                values["exit_price"] = exit_price
            if actual_pnl is not None:
                values["actual_pnl"] = actual_pnl

            await session.execute(
                update(Trade).where(Trade.id == trade_id).values(**values)
            )
            await session.commit()

        logger.info(f"[{self.name}] Trade {trade_id} status → {status}")

    async def save_patterns(self, trade_id: str, patterns: list[dict]):
        """Save detected patterns for a trade."""
        async with async_session_factory() as session:
            for p in patterns:
                pattern = Pattern(
                    trade_id=trade_id,
                    tag=p["tag"],
                    detail=p["detail"],
                    severity=p.get("severity", "medium"),
                )
                session.add(pattern)

            await session.commit()

        logger.info(f"[{self.name}] Saved {len(patterns)} patterns for trade {trade_id}")

    async def save_analysis_output(
        self,
        trade_id: str,
        heatmaps: list[dict],
        best_trade: dict,
        top_3: list[dict],
        metrics: dict,
        llm_coaching: str,
        execution_time_ms: float,
        total_simulations: int,
    ):
        """Save the final aggregated analysis output."""
        async with async_session_factory() as session:
            output = AnalysisOutput(
                trade_id=trade_id,
                heatmap_data=heatmaps,
                best_trade=best_trade,
                top_3=top_3,
                metrics=metrics,
                llm_coaching=llm_coaching,
                execution_time_ms=execution_time_ms,
                total_simulations=total_simulations,
            )
            session.add(output)
            await session.commit()

        logger.info(f"[{self.name}] Saved analysis output for trade {trade_id}")

    async def get_trade(self, trade_id: str) -> Optional[dict]:
        """Retrieve a trade by ID."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(Trade).where(Trade.id == trade_id)
            )
            trade = result.scalar_one_or_none()

            if trade:
                return {
                    "id": trade.id,
                    "asset": trade.asset,
                    "direction": trade.direction,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "size": trade.size,
                    "actual_pnl": trade.actual_pnl,
                    "status": trade.status,
                }

            return None

    async def get_analysis(self, trade_id: str) -> Optional[dict]:
        """Retrieve analysis output for a trade."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(AnalysisOutput).where(AnalysisOutput.trade_id == trade_id)
            )
            output = result.scalar_one_or_none()

            if output:
                return {
                    "heatmap_data": output.heatmap_data,
                    "best_trade": output.best_trade,
                    "top_3": output.top_3,
                    "metrics": output.metrics,
                    "llm_coaching": output.llm_coaching,
                    "execution_time_ms": output.execution_time_ms,
                    "total_simulations": output.total_simulations,
                }

            return None
