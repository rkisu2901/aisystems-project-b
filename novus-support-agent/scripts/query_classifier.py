"""
query_classifier.py — Tool router for the Novus Support Agent.

B1.3 deliverable: classify_tool() maps an incoming query to one of
four tool paths:

    policy_kb      — look up policy/product knowledge base
    order_tracker  — look up account/application status
    account_lookup — look up membership tier and benefits
    multi_tool     — requires both policy_kb AND order_tracker

Routing logic:
  1. LLM classifies intent (or caller passes pre-classified intent).
  2. Most intents map directly via INTENT_TO_TOOL (fast, deterministic).
  3. return_or_refund is ambiguous: it could be a general policy question
     (policy_kb) or a dispute on a specific account (multi_tool).
     A second LLM call disambiguates using the query text.

Usage:
    from scripts.query_classifier import classify_tool
    tool, intent = classify_tool("What is the foreclosure charge for Elite?")
    # → ("policy_kb", "return_or_refund")

    python scripts/query_classifier.py   # run 5 test queries covering all 4 paths
"""

from __future__ import annotations
import os, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

# ---------------------------------------------------------------------------
# B1.3 — Intent → Tool lookup table (fast path for unambiguous intents)
# ---------------------------------------------------------------------------

INTENT_TO_TOOL: dict[str, str] = {
    "order_status":       "order_tracker",   # account activation, disbursement timelines
    "product_info":       "policy_kb",        # interest rates, eligibility, features
    "membership":         "account_lookup",   # tier benefits, Elite/Plus/Standard
    "billing_or_payment": "policy_kb",        # EMI schedules, fee structure
    "general":            "policy_kb",        # catch-all: policy lookup
    # "return_or_refund" is intentionally absent — requires LLM disambiguation below
}

# ---------------------------------------------------------------------------
# LLM disambiguator for return_or_refund
# ---------------------------------------------------------------------------

DISAMBIGUATE_PROMPT = """A customer sent the following support query:

"{query}"

It has been classified as 'return_or_refund'. Determine which tool to use:

- policy_kb    : The customer is asking a general policy question about prepayment
                 terms, foreclosure rules, or refund eligibility. No specific account
                 dispute mentioned.
- multi_tool   : The customer is reporting a specific billing dispute, an incorrect
                 charge on their account, or requesting immediate reversal. Requires
                 both the policy knowledge base AND an account lookup.

Respond with ONLY one of: policy_kb  or  multi_tool"""


def _disambiguate_return(query: str) -> str:
    """Use LLM to decide if return_or_refund maps to policy_kb or multi_tool."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": DISAMBIGUATE_PROMPT.format(query=query),
        }],
    )
    result = response.choices[0].message.content.strip().lower()
    return result if result in ("policy_kb", "multi_tool") else "policy_kb"


# ---------------------------------------------------------------------------
# B1.3 — Main router: classify_tool()
# ---------------------------------------------------------------------------

def classify_tool(query: str, intent: str | None = None) -> tuple[str, str]:
    """Route a query to the appropriate tool.

    Args:
        query:  Raw customer query string.
        intent: Pre-classified intent (skip LLM call if already known).
                If None, classifies via the LLM classifier.

    Returns:
        (tool, intent) tuple.
        tool   ∈ {"policy_kb", "order_tracker", "account_lookup", "multi_tool"}
        intent ∈ the 6 standard intent classes

    Examples:
        classify_tool("What is the return window?")
        → ("policy_kb", "return_or_refund")

        classify_tool("Where is my order ORD-445521?")
        → ("order_tracker", "order_status")

        classify_tool("Am I Premium Gold?")
        → ("account_lookup", "membership")

        classify_tool("I'm Elite and was wrongly charged a penalty")
        → ("multi_tool", "return_or_refund")
    """
    if intent is None:
        from scripts.classifier_scratch import classify_llm
        intent = classify_llm(query)

    if intent == "return_or_refund":
        tool = _disambiguate_return(query)
    else:
        tool = INTENT_TO_TOOL.get(intent, "policy_kb")

    return tool, intent


# ---------------------------------------------------------------------------
# CLI — smoke test covering all 4 tool paths (B1.3 deliverable)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Path 1: policy_kb via return_or_refund (general question)
    ("What is the prepayment penalty for a personal loan?",   "policy_kb"),
    # Path 2: order_tracker
    ("How long does it take to activate my new savings account?", "order_tracker"),
    # Path 3: account_lookup
    ("Am I eligible for Elite tier benefits?",                "account_lookup"),
    # Path 4: multi_tool (specific dispute)
    ("I am Elite and have paid 20 EMIs but was charged a foreclosure penalty. Reverse it immediately.", "multi_tool"),
    # Path 1 again: policy_kb via billing_or_payment
    ("What are the processing fees for a personal loan?",     "policy_kb"),
]


if __name__ == "__main__":
    print("B1.3 — Tool Router: 5 test queries covering all 4 paths\n")
    print(f"{'Query':<58} {'Expected':>10}  {'Got':>10}  {'Pass?':>6}")
    print("-" * 90)

    all_pass = True
    for query, expected_tool in TEST_QUERIES:
        tool, intent = classify_tool(query)
        passed = "✓" if tool == expected_tool else "✗"
        if tool != expected_tool:
            all_pass = False
        print(f"{query[:57]:<58} {expected_tool:>10}  {tool:>10}  {passed:>6}  [{intent}]")

    print("-" * 90)
    print(f"\n{'All paths covered ✓' if all_pass else 'Some paths missed — check LLM disambiguator'}")
