"""
agent.py — 5-node LangGraph agent for the Novus Support Agent (P1C.1).

Replaces the naive handle_query() pipeline with a stateful graph that can:
  - Execute multi-step tool calls (multi_tool intent → 2 sequential calls)
  - Detect and escalate sensitive queries (billing disputes, fraud, SLA breaches)
  - Track trajectory: nodes visited, tools called, steps taken

Graph topology:
    classify_node → tool_node → evaluate_node ─┬→ respond_node → END
                         ↑                      ├→ tool_node  (loop for multi_tool)
                         └──────────────────────└→ escalate_node → END

Nodes:
    classify_node  — classify intent + route to tool via existing classify_tool()
    tool_node      — execute the assigned tool (policy_kb / order_tracker /
                     account_lookup); handles multi_tool 2-step loop
    evaluate_node  — decide next step: respond / loop / escalate
    respond_node   — assemble context from tool_results, generate final answer
    escalate_node  — generate escalation message, set should_escalate=True

Public entry point:
    run_agent(query: str) -> dict

Usage:
    python scripts/agent.py                    # 3 built-in demo trajectories
    python scripts/agent.py --query "..."      # single query with trajectory
    python scripts/agent.py --test             # run all 9 test queries
"""

from __future__ import annotations

import json
import operator
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ADR-8: Project A RAG API URL for the policy_kb tool path
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_MODEL    = "gpt-4o-mini"
TEMPERATURE     = 0.1
MAX_STEPS       = 6          # hard safety limit to prevent infinite loops

# Keyword patterns that strongly signal the query should be escalated to a
# human agent (billing disputes with evidence, security issues, SLA failures).
ESCALATION_KEYWORDS = [
    # Billing disputes / wrong charges
    "reverse immediately", "reversed immediately",
    "want this reversed", "get this reversed",
    "wrong charge", "wrongly charged", "incorrect charge",
    "wrong penalty", "wrongful",
    "dispute",
    # Fraud / security (both US and British spellings)
    "unauthorized", "unauthorised",
    "never authorized", "never authorised",
    "fraud",
    "account hacked", "security breach",
    "sim swap",
    # Urgency / distress signals
    "locked out",
    "life-threatening",
    "urgent", "urgently",
    "immediately",
    # Service failures / SLA breaches
    "pending for",
    "still not resolved",
    "multiple calls",
    "promised me",
    "despite multiple",
    "unacceptable delay", "sla breach",
    # Escalation intent
    "lodge complaint",
    "escalate",
    "legal action", "consumer court",
    # Duplicate / double charges
    "already paid", "charged twice", "double charged",
]

# ---------------------------------------------------------------------------
# AgentState — the shared state passed between all nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:          str
    intent:         str
    tool:           str
    # Annotated[list, operator.add] means LangGraph merges updates by appending
    tool_results:   Annotated[list[str], operator.add]
    final_answer:   str
    should_escalate: bool
    steps_taken:    int
    tools_called:   Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run_policy_kb(intent: str, query: str = "") -> str:
    """Retrieve policy context via Project A's RAG API (ADR-8).

    Calls POST /query on Project A for hybrid vector retrieval over the full
    19-doc corpus. Falls back to in-memory retrieval if the API is unreachable.
    """
    if query:
        try:
            resp = requests.post(
                f"{RAG_API_URL}/query",
                json={"query": query, "mode": "hybrid"},
                timeout=30,
            )
            resp.raise_for_status()
            ctx = resp.json().get("context", "")
            if ctx:
                return f"[policy_kb] {ctx}"
        except Exception:
            pass  # fall through to in-memory fallback

    try:
        from scripts.retrieval import retrieve_filtered, deduplicate_chunks
        chunks = retrieve_filtered(intent)
        unique, _ = deduplicate_chunks(chunks, threshold=0.75)
        context = "\n\n---\n\n".join(c["content"] for c in unique)
        return f"[policy_kb] {context}" if context else "[policy_kb] No relevant policy found."
    except Exception as e:
        return f"[policy_kb] Retrieval error: {e}"


def _run_order_tracker(query: str) -> str:
    """Mock order/activation status lookup.

    In production this would call a real order-management API.
    Returns a realistic mock response based on keywords in the query.
    """
    q = query.lower()
    if any(k in q for k in ("savings account", "saving", "account activation")):
        return (
            "[order_tracker] Savings account status: Application received. "
            "Standard activation is completed within 2 hours after Video KYC or branch verification. "
            "Physical debit card dispatched within 7 working days."
        )
    if any(k in q for k in ("personal loan", "loan", "disburse")):
        return (
            "[order_tracker] Personal loan status: Loan application under review. "
            "Disbursement timeline: within 24 hours for pre-approved customers; "
            "3–5 working days for non-pre-approved customers after final approval. "
            "Processing fee: 1% of loan amount (minimum Rs 999)."
        )
    return (
        "[order_tracker] Application status: Under processing. "
        "Estimated completion: 3–5 business days. "
        "Contact support with your application ID for real-time updates."
    )


def _run_account_lookup(query: str) -> str:
    """Mock account / membership tier lookup.

    In production this would query the core banking system.
    """
    q = query.lower()
    if "elite" in q:
        return (
            "[account_lookup] Account tier: Elite. "
            "Savings interest: 4.0% p.a. credited quarterly. "
            "Minimum balance: waived (no AQB requirement). "
            "ATM withdrawals: unlimited free at other bank ATMs. "
            "Personal loan rate: 11% p.a. "
            "Foreclosure: zero charge after 18 EMIs. "
            "Additional benefits: dedicated relationship manager, priority processing."
        )
    if "plus" in q or "novus plus" in q:
        return (
            "[account_lookup] Account tier: Novus Plus. "
            "Savings interest: 4.0% p.a. credited quarterly. "
            "Minimum balance: waived (AQB requirement: Rs 25,000). "
            "ATM withdrawals: 5 free per month at other bank ATMs. "
            "Personal loan rate: 13% p.a. "
            "Foreclosure: standard charges apply."
        )
    return (
        "[account_lookup] Account tier: Standard. "
        "Savings interest: 3.5% p.a. credited quarterly. "
        "Minimum balance: Rs 1,000. "
        "ATM withdrawals: 3 free per month at other bank ATMs. "
        "Personal loan rate: 16–18% p.a. "
        "Foreclosure: 1% charge after 12 EMIs."
    )


# ---------------------------------------------------------------------------
# Escalation detection
# ---------------------------------------------------------------------------

ESCALATION_JUDGE_PROMPT = """You are a banking support triage agent.
Determine if the following customer query requires immediate escalation to a
human agent, or can be handled automatically by the AI system.

Escalate when the query involves:
- A specific billing dispute or incorrect charge on the customer's account
- Account security, fraud, or unauthorised transactions
- Service delivery failures (significant SLA breach, wrong product delivered)
- Complex policy exceptions requiring managerial approval
- Threats of legal action or regulator complaints

Respond with ONLY one of: escalate  or  handle

Query: {query}
Tool results so far: {tool_summary}"""


def _should_escalate(query: str, tool_results: list[str]) -> bool:
    """Two-stage escalation check: fast keyword scan then LLM confirmation.

    Stage 1 (keyword, no API cost): any ESCALATION_KEYWORDS substring match.
    Stage 2 (LLM, only if stage 1 fires or query is ambiguous): GPT-4o-mini
    confirms whether this is a true escalation case.

    Returns True if the query should be routed to a human agent.
    """
    q_lower = query.lower()

    # Stage 1: fast keyword scan
    keyword_hit = any(kw in q_lower for kw in ESCALATION_KEYWORDS)

    if not keyword_hit:
        return False

    # Stage 2: LLM confirmation to reduce false positives
    tool_summary = " | ".join(tool_results[:2]) if tool_results else "none"
    # Use .replace() instead of .format() — user query may contain { or }
    # characters which would cause KeyError with str.format()
    prompt = (
        ESCALATION_JUDGE_PROMPT
        .replace("{query}", query)
        .replace("{tool_summary}", tool_summary[:500])
    )
    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = response.choices[0].message.content.strip().lower()
        return "escalate" in verdict
    except Exception:
        # If the LLM call fails, trust the keyword hit as a safe default
        return keyword_hit


# ---------------------------------------------------------------------------
# Node 1: classify_node
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> dict:
    """Classify intent and route to tool using the existing Week 2 classifier.

    Calls classify_tool() which runs:
      - classify_llm() for intent (GPT-4o-mini, temp=0)
      - INTENT_TO_TOOL lookup or _disambiguate_return() for tool routing

    Returns: intent, tool, steps_taken+1, tools_called += ["classify"]
    """
    from scripts.query_classifier import classify_tool
    tool, intent = classify_tool(state["query"])
    return {
        "intent":      intent,
        "tool":        tool,
        "steps_taken": state["steps_taken"] + 1,
        "tools_called": ["classify"],
    }


# ---------------------------------------------------------------------------
# Node 2: tool_node
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """Execute the appropriate tool and append its result to tool_results.

    Dispatch table:
        policy_kb      → _run_policy_kb(intent)
        order_tracker  → _run_order_tracker(query)
        account_lookup → _run_account_lookup(query)
        multi_tool     → first call: policy_kb; second call: account_lookup

    Multi-tool detection: if tool == "multi_tool" and tool_results already
    contains a policy_kb result, run account_lookup next.
    """
    tool        = state["tool"]
    intent      = state["intent"]
    query       = state["query"]
    n_results   = len(state["tool_results"])

    if tool == "policy_kb":
        result      = _run_policy_kb(intent, query=query)
        tool_label  = "policy_kb"

    elif tool == "order_tracker":
        result      = _run_order_tracker(query)
        tool_label  = "order_tracker"

    elif tool == "account_lookup":
        result      = _run_account_lookup(query)
        tool_label  = "account_lookup"

    elif tool == "multi_tool":
        if n_results == 0:
            # First pass: policy knowledge base via Project A RAG API
            result     = _run_policy_kb(intent, query=query)
            tool_label = "policy_kb"
        else:
            # Second pass: account lookup
            result     = _run_account_lookup(query)
            tool_label = "account_lookup"
    else:
        result      = _run_policy_kb(intent, query=query)
        tool_label  = "policy_kb_fallback"

    return {
        "tool_results":  [result],           # operator.add appends this
        "tools_called":  [tool_label],       # operator.add appends this
        "steps_taken":   state["steps_taken"] + 1,
    }


# ---------------------------------------------------------------------------
# Node 3: evaluate_node  (+ conditional edge router)
# ---------------------------------------------------------------------------

def evaluate_node(state: AgentState) -> dict:
    """Decide what happens next. Does NOT modify state — only reads it.

    The routing decision is made by _route_from_evaluate() below and is
    registered as a conditional edge on the graph. evaluate_node itself
    is a pass-through that exists so the conditional edge has a named node.
    """
    return {}   # state unchanged; routing is in _route_from_evaluate


def _route_from_evaluate(state: AgentState) -> str:
    """Conditional edge function: returns the name of the next node.

    Rules (evaluated in order):
      1. Safety: if steps_taken >= MAX_STEPS → "respond" (prevent runaway loops)
      2. Multi-tool loop: tool == "multi_tool" and only 1 result so far → "tool"
      3. Escalation: _should_escalate() → "escalate"
      4. Default → "respond"
    """
    if state["steps_taken"] >= MAX_STEPS:
        return "respond"

    if state["tool"] == "multi_tool" and len(state["tool_results"]) < 2:
        return "tool"

    if _should_escalate(state["query"], state["tool_results"]):
        return "escalate"

    return "respond"


# ---------------------------------------------------------------------------
# Node 4: respond_node
# ---------------------------------------------------------------------------

RESPOND_SYSTEM_PROMPT = """You are a helpful customer support agent for Novus Bank.

Answer the customer question using ONLY the information in the tool results below.
Tier-specific features (Standard, Plus, Elite) may appear inline — synthesize them
into a coherent answer when relevant.
If the answer is not present in the tool results, say:
"I don't have specific information about that. Please contact our support team."
Be concise, accurate, and professional."""


def respond_node(state: AgentState) -> dict:
    """Generate the final grounded answer from accumulated tool results.

    Assembles all tool_results into a single context block, then calls
    GPT-4o-mini with the system prompt + context + customer query.
    """
    context  = "\n\n---\n\n".join(state["tool_results"])
    messages = [
        {"role": "system", "content": RESPOND_SYSTEM_PROMPT + "\n\nTool Results:\n" + context},
        {"role": "user",   "content": state["query"]},
    ]
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=TEMPERATURE,
        messages=messages,
    )
    answer = response.choices[0].message.content.strip()
    return {
        "final_answer":   answer,
        "should_escalate": False,
        "steps_taken":    state["steps_taken"] + 1,
        "tools_called":   ["respond"],
    }


# ---------------------------------------------------------------------------
# Node 5: escalate_node
# ---------------------------------------------------------------------------

ESCALATE_SYSTEM_PROMPT = """You are a customer support agent for Novus Bank.
This query has been flagged for escalation to a human agent.

Generate a professional acknowledgement message that:
1. Confirms you have noted the customer's concern
2. Tells them a human agent will contact them within 2 business hours
3. Provides the escalation reference format: ESC-<timestamp>
Be empathetic, brief, and professional."""


def escalate_node(state: AgentState) -> dict:
    """Generate an escalation acknowledgement and flag should_escalate=True."""
    ts  = int(time.time())
    ref = f"ESC-{ts}"
    messages = [
        {"role": "system", "content": ESCALATE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Query: {state['query']}\nEscalation ref: {ref}"},
    ]
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=TEMPERATURE,
        messages=messages,
    )
    answer = response.choices[0].message.content.strip()
    return {
        "final_answer":    answer,
        "should_escalate": True,
        "steps_taken":     state["steps_taken"] + 1,
        "tools_called":    ["escalate"],
    }


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("classify",  classify_node)
    workflow.add_node("execute",   tool_node)    # "tool" clashes with AgentState.tool key
    workflow.add_node("evaluate",  evaluate_node)
    workflow.add_node("respond",   respond_node)
    workflow.add_node("escalate",  escalate_node)

    # Fixed edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify",  "execute")
    workflow.add_edge("execute",   "evaluate")
    workflow.add_edge("respond",   END)
    workflow.add_edge("escalate",  END)

    # Conditional edge from evaluate
    workflow.add_conditional_edges(
        "evaluate",
        _route_from_evaluate,
        {
            "tool":     "execute",   # loop back to execute node
            "respond":  "respond",
            "escalate": "escalate",
        },
    )

    return workflow.compile()


# Module-level compiled graph (singleton)
app = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(query: str) -> dict:
    """Run the LangGraph agent on a customer query.

    Returns a dict with the same keys as handle_query() plus agent-specific
    fields (steps_taken, tools_called) for trajectory analysis.

    Return shape:
        {
            "query":         str,
            "intent":        str,    one of 6 intent classes
            "tool":          str,    initial tool route assigned by classify_node
            "escalation":    bool,   True if escalated to human agent
            "answer":        str,    final grounded answer or escalation message
            "context":       str,    concatenated tool results (for eval harness)
            "trace_id":      None,   stub — LangFuse integration future work
            "steps_taken":   int,    total node executions (classify + tools + respond/escalate)
            "tools_called":  list,   ordered list of tool labels executed
        }
    """
    initial_state: AgentState = {
        "query":          query,
        "intent":         "",
        "tool":           "",
        "tool_results":   [],
        "final_answer":   "",
        "should_escalate": False,
        "steps_taken":    0,
        "tools_called":   [],
    }

    try:
        result = app.invoke(initial_state)
    except Exception as e:
        # Return a safe error dict so the eval harness can continue across
        # all 32 golden dataset entries even if one query fails
        return {
            "query":        query,
            "intent":       "unknown",
            "tool":         "unknown",
            "escalation":   False,
            "answer":       f"Agent error: {e}",
            "context":      "",
            "trace_id":     None,
            "steps_taken":  0,
            "tools_called": [],
        }

    return {
        "query":        result["query"],
        "intent":       result["intent"],
        "tool":         result["tool"],
        "escalation":   result["should_escalate"],
        "answer":       result["final_answer"],
        "context":      "\n\n---\n\n".join(result["tool_results"]),
        "trace_id":     None,
        "steps_taken":  result["steps_taken"],
        "tools_called": result["tools_called"],
    }


# ---------------------------------------------------------------------------
# CLI — demo trajectories (P1C.1 deliverable)
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    # Trajectory 1: Simple — 1 tool call
    ("What is the minimum balance for a Novus Plus savings account?",
     "Expected: 1 tool call (account_lookup), no escalation"),

    # Trajectory 2: Complex — 2 tool calls (multi_tool)
    ("I am an Elite customer and I believe I was wrongly charged a "
     "foreclosure penalty after paying 15 EMIs. Please reverse it immediately.",
     "Expected: 2 tool calls (policy_kb + account_lookup), escalation=True"),

    # Trajectory 3: Escalation case
    ("Someone has made an unauthorized transaction on my account. "
     "This is fraud — escalate this immediately.",
     "Expected: 1-2 tool calls, escalation=True"),
]

TEST_QUERIES = [
    "What is the minimum balance for a savings account?",
    "Can I prepay my personal loan early?",
    "What are the benefits of Elite membership?",
    "How long does it take to activate my savings account?",
    "What is the processing fee for a personal loan?",
    "I am an Elite customer charged a wrong penalty — reverse it.",
    "What is the interest rate on a personal loan?",
    "What is the AQB for Novus Plus membership?",
    "Can I foreclose after 10 EMIs?",
]


def _print_trajectory(query: str, result: dict, note: str = "", elapsed: float = 0.0) -> None:
    print(f"\n{'='*65}")
    print(f"Query      : {query}")
    if note:
        print(f"Note       : {note}")
    print(f"Intent     : {result['intent']}  →  Tool: {result['tool']}")
    print(f"Escalate   : {result['escalation']}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Tools called: {result['tools_called']}")
    print(f"Answer     : {result['answer'][:200]}{'…' if len(result['answer']) > 200 else ''}")
    print(f"Time       : {elapsed:.2f}s")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Novus Support LangGraph Agent")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--test",  action="store_true", help="Run all 9 test queries")
    args = parser.parse_args()

    if args.test:
        print("\nRunning all 9 test queries through LangGraph agent...\n")
        for q in TEST_QUERIES:
            t0 = time.time()
            result = run_agent(q)
            _print_trajectory(q, result, elapsed=round(time.time() - t0, 2))
        return

    if args.query:
        t0 = time.time()
        result = run_agent(args.query)
        _print_trajectory(args.query, result, elapsed=round(time.time() - t0, 2))
        return

    # Default: 3 demo trajectories (P1C.1 deliverable)
    print("\nP1C.1 — LangGraph Agent: 3 demo trajectories\n")
    for query, note in DEMO_QUERIES:
        t0 = time.time()
        result = run_agent(query)
        _print_trajectory(query, result, note=note, elapsed=round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
