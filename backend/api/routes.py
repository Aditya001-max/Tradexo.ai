"""
API Routes
==========
FastAPI router exposing trade submission and result retrieval endpoints.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config import get_settings
from backend.database import get_db
from backend.models import Trade, AnalysisOutput, Pattern
from backend.schemas import (
    TradeInput,
    TradeSubmitResponse,
    AnalysisResultResponse,
    HealthResponse,
    ActualTradeDetail,
    SimulationDetail,
    HeatmapData,
    PatternDetail,
    AnalysisMetrics,
)
from backend.utils.logger import get_logger

logger = get_logger("api")
settings = get_settings()

router = APIRouter()


# ============================================
# POST /submit-trade
# ============================================
@router.post("/submit-trade", response_model=TradeSubmitResponse, status_code=202)
async def submit_trade(
    trade_input: TradeInput,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a trade for counterfactual analysis.
    Returns immediately with a trade_id. Processing happens in background.
    """
    logger.info(
        "Trade submitted",
        extra={
            "asset": trade_input.asset,
            "direction": trade_input.direction,
            "entry_time": str(trade_input.entry_time),
            "exit_time": str(trade_input.exit_time),
        },
    )

    # --- Create trade record ---
    trade = Trade(
        asset=trade_input.asset,
        direction=trade_input.direction,
        entry_time=trade_input.entry_time,
        exit_time=trade_input.exit_time,
        size=trade_input.size,
        status="pending",
    )
    db.add(trade)
    await db.flush()
    trade_id = trade.id
    await db.commit()

    logger.info(f"Trade created with id={trade_id}")

    # --- Trigger background analysis ---
    background_tasks.add_task(_run_pipeline, trade_id, trade_input)

    return TradeSubmitResponse(
        trade_id=trade_id,
        status="pending",
        message=f"Trade submitted. Poll GET /results/{trade_id} for results.",
    )


async def _run_pipeline(trade_id: str, trade_input: TradeInput):
    """
    Execute the full analysis pipeline in the background.
    Imported lazily to avoid circular imports.
    """
    from backend.agents.orchestrator import run_analysis_pipeline

    try:
        await run_analysis_pipeline(trade_id, trade_input)
    except Exception as e:
        logger.error(f"Pipeline failed for trade {trade_id}: {e}", exc_info=True)


# ============================================
# GET /results/{trade_id}
# ============================================
@router.get("/results/{trade_id}", response_model=AnalysisResultResponse)
async def get_results(trade_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieve analysis results for a submitted trade.
    Returns status='processing' if analysis is still running.
    """
    # --- Fetch trade ---
    result = await db.execute(select(Trade).where(Trade.id == trade_id))
    trade = result.scalar_one_or_none()

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # --- If still processing ---
    if trade.status in ("pending", "processing"):
        return AnalysisResultResponse(
            trade_id=trade_id,
            status=trade.status,
        )

    # --- If failed ---
    if trade.status == "failed":
        return AnalysisResultResponse(
            trade_id=trade_id,
            status="failed",
            error_message=trade.error_message,
        )

    # --- Fetch analysis output ---
    analysis_result = await db.execute(
        select(AnalysisOutput).where(AnalysisOutput.trade_id == trade_id)
    )
    analysis = analysis_result.scalar_one_or_none()

    # --- Fetch patterns ---
    patterns_result = await db.execute(
        select(Pattern).where(Pattern.trade_id == trade_id)
    )
    patterns = patterns_result.scalars().all()

    # --- Build response ---
    response = AnalysisResultResponse(
        trade_id=trade_id,
        status="completed",
        actual_trade=ActualTradeDetail(
            asset=trade.asset,
            direction=trade.direction,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            size=trade.size,
            pnl=trade.actual_pnl,
        ),
    )

    if analysis:
        # --- Heatmap ---
        if analysis.heatmap_data:
            response.heatmap = [
                HeatmapData(**hm) for hm in analysis.heatmap_data
            ]

        # --- Best trade ---
        if analysis.best_trade:
            response.best_trade = SimulationDetail(**analysis.best_trade)

        # --- Top 3 ---
        if analysis.top_3:
            response.top_3 = [SimulationDetail(**t) for t in analysis.top_3]

        # --- Metrics ---
        if analysis.metrics:
            response.metrics = AnalysisMetrics(**analysis.metrics)

        # --- LLM coaching ---
        response.llm_coaching = analysis.llm_coaching

    # --- Patterns ---
    if patterns:
        response.patterns = [
            PatternDetail(tag=p.tag, detail=p.detail, severity=p.severity)
            for p in patterns
        ]

    return response


# ============================================
# GET /health
# ============================================
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        project=settings.PROJECT_NAME,
    )
