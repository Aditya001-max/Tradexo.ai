"""
Pydantic Schemas
================
Request/Response schemas for the API layer.
These define the data contracts between the API and the agents.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================
# REQUEST SCHEMAS
# ============================================

class TradeInput(BaseModel):
    """Input schema for submitting a trade for analysis."""

    asset: str = Field(
        ...,
        description="Ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')",
        min_length=1,
        max_length=20,
        examples=["AAPL"],
    )
    direction: str = Field(
        ...,
        description="Trade direction: 'long' or 'short'",
        examples=["long"],
    )
    entry_time: datetime = Field(
        ...,
        description="Trade entry timestamp (ISO 8601)",
        examples=["2026-04-20T10:30:00"],
    )
    exit_time: datetime = Field(
        ...,
        description="Trade exit timestamp (ISO 8601)",
        examples=["2026-04-20T14:00:00"],
    )
    size: float = Field(
        ...,
        description="Position size (number of shares/units)",
        gt=0,
        examples=[100],
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("long", "short"):
            raise ValueError("Direction must be 'long' or 'short'")
        return v

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("exit_time")
    @classmethod
    def validate_exit_after_entry(cls, v: datetime, info) -> datetime:
        entry = info.data.get("entry_time")
        if entry and v <= entry:
            raise ValueError("exit_time must be after entry_time")
        return v


# ============================================
# RESPONSE SCHEMAS
# ============================================

class TradeSubmitResponse(BaseModel):
    """Response after submitting a trade."""
    trade_id: str
    status: str = "pending"
    message: str = "Trade submitted for analysis. Poll /results/{trade_id} for results."


class ActualTradeDetail(BaseModel):
    """Details of the actual trade as executed."""
    asset: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    size: float
    pnl: Optional[float] = None


class SimulationDetail(BaseModel):
    """Details of a single counterfactual simulation."""
    entry_shift: int
    exit_shift: int
    stop_loss: float
    size_multiplier: float
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: float
    drawdown: float
    exit_reason: str
    improvement: Optional[str] = None


class HeatmapData(BaseModel):
    """Heatmap data for visualization."""
    entry_shifts: list[int]
    exit_shifts: list[int]
    pnl_matrix: list[list[float]]
    stop_loss: float = Field(description="Stop loss level this heatmap is for")
    size_multiplier: float = Field(description="Size multiplier this heatmap is for")


class PatternDetail(BaseModel):
    """A detected behavioral pattern."""
    tag: str
    detail: str
    severity: str = "medium"


class AnalysisMetrics(BaseModel):
    """Summary metrics of the analysis."""
    total_simulations: int
    execution_time_ms: float
    best_pnl: float
    worst_pnl: float
    median_pnl: float
    actual_pnl: Optional[float] = None
    improvement_over_actual: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    """Full analysis result returned to the user."""
    trade_id: str
    status: str
    actual_trade: Optional[ActualTradeDetail] = None
    heatmap: Optional[list[HeatmapData]] = None
    best_trade: Optional[SimulationDetail] = None
    top_3: Optional[list[SimulationDetail]] = None
    patterns: Optional[list[PatternDetail]] = None
    llm_coaching: Optional[str] = None
    metrics: Optional[AnalysisMetrics] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    project: str
