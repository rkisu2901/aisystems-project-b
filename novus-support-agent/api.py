"""
api.py — FastAPI HTTP wrapper for the Novus Support Agent (LangGraph agent).

Endpoints:
  GET  /health  — ALB health check
  POST /query   — run LangGraph agent, return answer + trajectory

Usage (local):
  uvicorn api:app --host 0.0.0.0 --port 8080
  curl localhost:8080/health
  curl -X POST localhost:8080/query \
       -H 'Content-Type: application/json' \
       -d '{"query": "What is the minimum balance for Novus Plus?"}'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Ensure scripts/ is importable from /app (WORKDIR in Dockerfile)
sys.path.insert(0, str(Path(__file__).parent))

import time
from collections import defaultdict, deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Novus Support Agent", version="1.0.0")

_FRONTEND = Path(__file__).parent / "frontend" / "index.html"

# ---------------------------------------------------------------------------
# Session-level stats tracker (in-memory, resets on restart)
# ---------------------------------------------------------------------------

_stats: dict = {
    "total_queries":      0,
    "escalated":          0,
    "auto_handled":       0,
    "multi_tool_queries": 0,
    "total_steps":        0,
    "total_elapsed":      0.0,
    "total_ctx_length":   0,
    "tool_counts":        defaultdict(int),
    "intent_counts":      defaultdict(int),
    "trajectory_counts":  defaultdict(int),
    "steps_distribution": defaultdict(int),
    # {intent: {"escalated": N, "auto": N}}
    "escalation_by_intent": defaultdict(lambda: {"escalated": 0, "auto": 0}),
    # rolling window for queries-per-minute
    "query_timestamps":   deque(maxlen=300),
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    tool: str
    escalation: bool
    steps_taken: int
    tools_called: list[str]
    context: str
    elapsed_seconds: float
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def frontend():
    return FileResponse(_FRONTEND)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the LangGraph agent on the customer query and return the result."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.time()
    try:
        from scripts.agent import run_agent
        result = run_agent(request.query.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    elapsed = round(time.time() - t0, 2)

    # record session stats
    intent     = result["intent"]
    tools      = result["tools_called"]
    steps      = result["steps_taken"]
    ctx_length = len(result.get("context", ""))
    escalated  = result["escalation"]
    trajectory = " → ".join(tools)

    _stats["total_queries"]      += 1
    _stats["total_steps"]        += steps
    _stats["total_elapsed"]      += elapsed
    _stats["total_ctx_length"]   += ctx_length
    _stats["steps_distribution"][steps] += 1
    _stats["trajectory_counts"][trajectory] += 1
    _stats["query_timestamps"].append(time.time())

    if escalated:
        _stats["escalated"] += 1
        _stats["escalation_by_intent"][intent]["escalated"] += 1
    else:
        _stats["auto_handled"] += 1
        _stats["escalation_by_intent"][intent]["auto"] += 1

    if result.get("tool") == "multi_tool":
        _stats["multi_tool_queries"] += 1

    for tool in tools:
        _stats["tool_counts"][tool] += 1
    _stats["intent_counts"][intent] += 1

    return QueryResponse(
        answer=result["answer"],
        intent=result["intent"],
        tool=result["tool"],
        escalation=result["escalation"],
        steps_taken=result["steps_taken"],
        tools_called=result["tools_called"],
        context=result["context"],
        elapsed_seconds=elapsed,
        trace_id=result.get("trace_id"),
    )


@app.get("/stats")
def get_stats():
    total = _stats["total_queries"]
    # queries per minute: count timestamps in the last 60 seconds
    now = time.time()
    recent = sum(1 for t in _stats["query_timestamps"] if now - t <= 60)
    return {
        "total_queries":        total,
        "auto_handled":         _stats["auto_handled"],
        "escalated":            _stats["escalated"],
        "escalation_rate":      round(_stats["escalated"] / total, 4) if total else 0.0,
        "avg_steps":            round(_stats["total_steps"] / total, 2) if total else 0.0,
        "avg_elapsed_seconds":  round(_stats["total_elapsed"] / total, 2) if total else 0.0,
        "avg_context_length":   round(_stats["total_ctx_length"] / total) if total else 0,
        "multi_tool_rate":      round(_stats["multi_tool_queries"] / total, 4) if total else 0.0,
        "queries_per_minute":   round(recent, 2),
        "tool_counts":          dict(_stats["tool_counts"]),
        "intent_counts":        dict(_stats["intent_counts"]),
        "trajectory_counts":    dict(sorted(_stats["trajectory_counts"].items(), key=lambda x: -x[1])[:5]),
        "steps_distribution":   {str(k): v for k, v in sorted(_stats["steps_distribution"].items())},
        "escalation_by_intent": {k: dict(v) for k, v in _stats["escalation_by_intent"].items()},
    }


@app.post("/stats/reset")
def reset_stats():
    _stats["total_queries"]      = 0
    _stats["escalated"]          = 0
    _stats["auto_handled"]       = 0
    _stats["multi_tool_queries"] = 0
    _stats["total_steps"]        = 0
    _stats["total_elapsed"]      = 0.0
    _stats["total_ctx_length"]   = 0
    _stats["tool_counts"].clear()
    _stats["intent_counts"].clear()
    _stats["trajectory_counts"].clear()
    _stats["steps_distribution"].clear()
    _stats["escalation_by_intent"].clear()
    _stats["query_timestamps"].clear()
    return {"status": "reset"}
