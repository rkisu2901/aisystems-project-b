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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Novus Support Agent", version="1.0.0")


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
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the LangGraph agent on the customer query and return the result."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        from scripts.agent import run_agent
        result = run_agent(request.query.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return QueryResponse(
        answer=result["answer"],
        intent=result["intent"],
        tool=result["tool"],
        escalation=result["escalation"],
        steps_taken=result["steps_taken"],
        tools_called=result["tools_called"],
        context=result["context"],
        trace_id=result.get("trace_id"),
    )
