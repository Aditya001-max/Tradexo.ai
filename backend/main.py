"""
FastAPI Application Entry Point
================================
Counterfactual Trade Analysis Engine — Main Application.

Starts the FastAPI server with:
- CORS middleware
- Database initialization on startup
- API router mounting
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db
from backend.api.routes import router
from backend.utils.logger import get_logger

settings = get_settings()
logger = get_logger("main")


# ============================================
# LIFESPAN: Startup / Shutdown
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup."""
    logger.info("🚀 Starting Counterfactual Trade Analysis Engine")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("👋 Shutting down")


# ============================================
# APP INSTANCE
# ============================================
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Multi-agent counterfactual trade analysis engine. "
        "Submit a trade, run 1,600+ parallel simulations, "
        "and receive behavioral coaching powered by LLMs."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Router ---
app.include_router(router, prefix="/api/v1", tags=["Trade Analysis"])


# ============================================
# RUN DIRECTLY
# ============================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
