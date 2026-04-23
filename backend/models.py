"""
ORM Models
==========
SQLAlchemy ORM models for trades, simulation results, and patterns.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Trade(Base):
    """Stores submitted trades."""

    __tablename__ = "trades"

    id = Column(String, primary_key=True, default=generate_uuid)
    asset = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'long' or 'short'
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=True)   # filled by market data agent
    exit_price = Column(Float, nullable=True)     # filled by market data agent
    size = Column(Float, nullable=False)
    actual_pnl = Column(Float, nullable=True)
    status = Column(String(20), default="pending")  # pending | processing | completed | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)

    # --- Relationships ---
    results = relationship("SimulationResult", back_populates="trade", cascade="all, delete-orphan")
    patterns = relationship("Pattern", back_populates="trade", cascade="all, delete-orphan")
    analysis = relationship("AnalysisOutput", back_populates="trade", uselist=False, cascade="all, delete-orphan")


class SimulationResult(Base):
    """Stores individual counterfactual simulation results."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=False, index=True)

    # --- Parameters ---
    entry_shift = Column(Integer, nullable=False)       # minutes
    exit_shift = Column(Integer, nullable=False)         # minutes
    stop_loss = Column(Float, nullable=False)            # e.g., 0.01 = 1%
    size_multiplier = Column(Float, nullable=False)      # e.g., 1.5 = 150%

    # --- Outcomes ---
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    exit_reason = Column(String(20), nullable=False)     # 'time' | 'stop_loss'

    # --- Relationship ---
    trade = relationship("Trade", back_populates="results")


class Pattern(Base):
    """Stores detected behavioral patterns/mistakes."""

    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=False, index=True)
    tag = Column(String(50), nullable=False)         # e.g., 'EARLY_ENTRY'
    detail = Column(Text, nullable=False)
    severity = Column(String(10), default="medium")  # low | medium | high

    # --- Relationship ---
    trade = relationship("Trade", back_populates="patterns")


class AnalysisOutput(Base):
    """Stores the final aggregated analysis output for a trade."""

    __tablename__ = "analysis_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=False, unique=True, index=True)

    # --- Aggregated Data (stored as JSON) ---
    heatmap_data = Column(JSON, nullable=True)
    best_trade = Column(JSON, nullable=True)
    top_3 = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    llm_coaching = Column(Text, nullable=True)

    execution_time_ms = Column(Float, nullable=True)
    total_simulations = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # --- Relationship ---
    trade = relationship("Trade", back_populates="analysis")
