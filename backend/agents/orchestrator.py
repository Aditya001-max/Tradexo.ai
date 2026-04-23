"""
Orchestrator
============
Pipeline coordinator that chains all agents together.

Workflow:
1. Ingestion Agent → validates trade
2. Market Data Agent → fetches OHLCV
3. Simulation Agent → generates parameter grid
4. Parallel Agent → runs counterfactual simulations
5. Aggregation Agent → builds heatmaps + rankings
6. Pattern Agent → detects behavioral mistakes
7. LLM Agent → generates coaching
8. Storage Agent → persists everything

This is the brain of the system — it coordinates all agents
and handles errors at each stage.
"""

import asyncio
import time
from typing import Any

from backend.schemas import TradeInput
from backend.agents.ingestion_agent import IngestionAgent
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.simulation_agent import SimulationAgent
from backend.agents.parallel_agent import ParallelAgent
from backend.agents.aggregation_agent import AggregationAgent
from backend.agents.pattern_agent import PatternDetectionAgent
from backend.agents.llm_agent import LLMAgent
from backend.agents.storage_agent import StorageAgent
from backend.utils.logger import get_logger

logger = get_logger("orchestrator")


async def run_analysis_pipeline(trade_id: str, trade_input: TradeInput):
    """
    Execute the full counterfactual analysis pipeline.

    This is the master workflow that coordinates all 9 agents
    in sequence. Each stage feeds into the next.

    Args:
        trade_id: The database ID of the trade
        trade_input: The validated trade input from the API

    Raises:
        Exception: Any unhandled error (caught and stored by the API layer)
    """
    pipeline_start = time.perf_counter()
    storage = StorageAgent()

    logger.info(f"🚀 Starting analysis pipeline for trade {trade_id}")

    try:
        # ============================================
        # STAGE 1: INGESTION — Validate & Normalize
        # ============================================
        await storage.update_trade_status(trade_id, "processing")
        logger.info(f"[Pipeline] Stage 1/7: Ingestion")

        ingestion = IngestionAgent()
        clean_trade = ingestion.validate_and_normalize(trade_input)
        clean_trade["id"] = trade_id  # Attach DB id

        logger.info(f"[Pipeline] ✅ Trade validated: {clean_trade['asset']} {clean_trade['direction']}")

        # ============================================
        # STAGE 2: MARKET DATA — Fetch OHLCV
        # ============================================
        logger.info(f"[Pipeline] Stage 2/7: Fetching market data")

        market_agent = MarketDataAgent()

        # Run yfinance fetch in thread pool (it's synchronous / blocking)
        loop = asyncio.get_event_loop()
        market_data = await loop.run_in_executor(
            None,
            market_agent.fetch,
            clean_trade["asset"],
            clean_trade["entry_time"],
            clean_trade["exit_time"],
        )

        # Get actual entry/exit prices
        entry_price = market_agent.get_price_at_time(market_data, clean_trade["entry_time"])
        exit_price = market_agent.get_price_at_time(market_data, clean_trade["exit_time"])

        # Compute actual P&L
        actual_pnl = 0.0
        if entry_price and exit_price:
            if clean_trade["direction"] == "long":
                actual_pnl = (exit_price - entry_price) * clean_trade["size"]
            else:
                actual_pnl = (entry_price - exit_price) * clean_trade["size"]

        # Update trade with prices
        await storage.update_trade_status(
            trade_id, "processing",
            entry_price=entry_price,
            exit_price=exit_price,
            actual_pnl=actual_pnl,
        )

        clean_trade["entry_price"] = entry_price
        clean_trade["exit_price"] = exit_price
        clean_trade["actual_pnl"] = actual_pnl

        logger.info(
            f"[Pipeline] ✅ Market data: {len(market_data)} bars | "
            f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} | "
            f"Actual P&L: ${actual_pnl:.2f}"
        )

        # ============================================
        # STAGE 3: SIMULATION — Generate Parameter Grid
        # ============================================
        logger.info(f"[Pipeline] Stage 3/7: Generating parameter grid")

        sim_agent = SimulationAgent()
        param_grid = sim_agent.generate_param_grid(clean_trade)

        logger.info(f"[Pipeline] ✅ Generated {len(param_grid)} parameter sets")

        # ============================================
        # STAGE 4: PARALLEL EXECUTION — Run Simulations
        # ============================================
        logger.info(f"[Pipeline] Stage 4/7: Running parallel simulations")

        parallel = ParallelAgent()

        # Run in executor since ProcessPoolExecutor is blocking
        try:
            results, exec_time = await loop.run_in_executor(
                None,
                parallel.run_counterfactuals,
                param_grid,
                market_data,
            )
        except Exception as e:
            logger.warning(f"[Pipeline] Parallel execution failed, using fallback: {e}")
            results, exec_time = await loop.run_in_executor(
                None,
                parallel.run_sequential_fallback,
                param_grid,
                market_data,
            )

        logger.info(
            f"[Pipeline] ✅ {len(results)} simulations completed in {exec_time:.1f}ms"
        )

        # ============================================
        # STAGE 5: AGGREGATION — Build Heatmaps & Rankings
        # ============================================
        logger.info(f"[Pipeline] Stage 5/7: Aggregating results")

        agg_agent = AggregationAgent()
        aggregated = agg_agent.aggregate(results, actual_pnl=actual_pnl)

        best_trade = aggregated.get("best_trade")
        top_3 = aggregated.get("top_3", [])
        heatmaps = aggregated.get("heatmaps", [])
        metrics = aggregated.get("metrics", {})

        # Add execution time to metrics
        metrics["execution_time_ms"] = round(exec_time, 1)

        logger.info(
            f"[Pipeline] ✅ Best P&L: ${metrics.get('best_pnl', 0):.2f} | "
            f"Median: ${metrics.get('median_pnl', 0):.2f} | "
            f"Profitable: {metrics.get('profitable_pct', 0):.0f}%"
        )

        # ============================================
        # STAGE 6: PATTERN DETECTION — Tag Behavioral Mistakes
        # ============================================
        logger.info(f"[Pipeline] Stage 6/7: Detecting patterns")

        pattern_agent = PatternDetectionAgent()
        patterns = pattern_agent.detect_patterns(clean_trade, best_trade, aggregated)

        logger.info(
            f"[Pipeline] ✅ Detected {len(patterns)} patterns: "
            f"{[p['tag'] for p in patterns]}"
        )

        # ============================================
        # STAGE 7: LLM COACHING — Generate Insights
        # ============================================
        logger.info(f"[Pipeline] Stage 7/7: Generating LLM coaching")

        llm = LLMAgent()
        coaching = llm.generate_coaching(
            trade=clean_trade,
            patterns=patterns,
            best_trade=best_trade or {},
            metrics=metrics,
        )

        logger.info(f"[Pipeline] ✅ Coaching generated ({len(coaching)} chars)")

        # ============================================
        # PERSIST EVERYTHING
        # ============================================
        logger.info(f"[Pipeline] Saving results to database...")

        # Save patterns
        await storage.save_patterns(trade_id, patterns)

        # Save analysis output
        await storage.save_analysis_output(
            trade_id=trade_id,
            heatmaps=heatmaps,
            best_trade=best_trade,
            top_3=top_3,
            metrics=metrics,
            llm_coaching=coaching,
            execution_time_ms=exec_time,
            total_simulations=len(results),
        )

        # Mark trade as completed
        await storage.update_trade_status(trade_id, "completed")

        pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"🏁 Pipeline completed for trade {trade_id} in {pipeline_elapsed:.0f}ms | "
            f"{len(results)} simulations | {len(patterns)} patterns"
        )

    except ValueError as e:
        # Validation errors
        logger.error(f"[Pipeline] Validation error: {e}")
        await storage.update_trade_status(trade_id, "failed", error_message=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"[Pipeline] Unexpected error: {e}", exc_info=True)
        await storage.update_trade_status(
            trade_id, "failed",
            error_message=f"Pipeline error: {str(e)}"
        )
