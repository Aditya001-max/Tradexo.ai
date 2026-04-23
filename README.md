# 🔬 Counterfactual Trade Analysis Engine

A production-grade, multi-agent system that analyzes trades by running **1,600+ parallel counterfactual simulations** and generating **AI-powered behavioral coaching**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (API Agent)                    │
├────────┬────────┬──────────┬──────────┬────────┬────────┤
│Ingest  │Market  │Simulation│ Parallel │Pattern │  LLM   │
│Agent   │Data    │Agent     │ Agent    │Agent   │ Agent  │
│        │Agent   │          │(Workers) │        │(HF API)│
├────────┴────────┴──────────┴──────────┴────────┴────────┤
│              Storage Agent (SQLite/PostgreSQL)            │
└─────────────────────────────────────────────────────────┘
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd "Quant project"
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your HF_API_TOKEN (optional — system works without it)
```

### 3. Start the Server
```bash
python -m backend.main
```

### 4. Submit a Trade
```bash
curl -X POST http://localhost:8000/api/v1/submit-trade \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "AAPL",
    "direction": "long",
    "entry_time": "2026-04-20T10:30:00",
    "exit_time": "2026-04-20T14:00:00",
    "size": 100
  }'
```

### 5. Get Results
```bash
curl http://localhost:8000/api/v1/results/{trade_id}
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/submit-trade` | Submit a trade for analysis |
| GET | `/api/v1/results/{trade_id}` | Get analysis results |
| GET | `/api/v1/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

## 🤖 Agents

| # | Agent | Role |
|---|-------|------|
| 1 | **Ingestion Agent** | Validates & normalizes trade input |
| 2 | **Market Data Agent** | Fetches OHLCV data via Yahoo Finance |
| 3 | **Simulation Agent** | Generates parameter grid (1,620 combos) |
| 4 | **Parallel Agent** | Distributes simulations across CPU cores |
| 5 | **Aggregation Agent** | Builds heatmaps, rankings, metrics |
| 6 | **Pattern Agent** | Detects behavioral mistakes |
| 7 | **LLM Agent** | Generates coaching via HuggingFace |
| 8 | **API Agent** | FastAPI endpoints |
| 9 | **Storage Agent** | Database persistence |

## 📊 Output Format

```json
{
  "trade_id": "uuid",
  "status": "completed",
  "actual_trade": { ... },
  "heatmap": [{ "entry_shifts": [...], "exit_shifts": [...], "pnl_matrix": [[...]] }],
  "best_trade": { "pnl": 450.00, "improvement": "+246%" },
  "top_3": [...],
  "patterns": [{ "tag": "EARLY_ENTRY", "detail": "..." }],
  "llm_coaching": "Based on your trade...",
  "metrics": { "total_simulations": 1620, "profitable_pct": 73.5 }
}
```

## 🔧 Tech Stack

- **Backend:** FastAPI + Python 3.11+
- **Parallel Processing:** concurrent.futures (ProcessPoolExecutor)
- **Data:** pandas, numpy, yfinance
- **Database:** SQLite (swappable to PostgreSQL)
- **LLM:** HuggingFace Inference API
- **Caching:** In-memory TTL cache (swappable to Redis)
