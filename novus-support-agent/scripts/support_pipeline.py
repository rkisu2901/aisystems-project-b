"""
support_pipeline.py — Novus Support Agent query-handling pipeline.

Three-step pipeline per query:
  1. classify_intent()  — GPT-4o-mini zero-shot classifier (6 intent classes)
  2. route()            — Week 1 baseline: always handle, never escalate (0% catch rate)
  3. generate_answer()  — GPT-4o-mini grounded on data/products/*.md (in-memory)

Public entry point:
    handle_query(query: str) -> dict

Usage:
    python scripts/support_pipeline.py                   # 3 built-in test queries
    python scripts/support_pipeline.py --query "..."     # single query
"""

import argparse
import os
import sys
import time

# Windows console defaults to cp1252 which can't print ₹ — force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_ENABLED = True
except Exception as e:
    _lf = None
    LANGFUSE_ENABLED = False
    print(f"[LangFuse] disabled: {e}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLASSIFY_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
ESCALATION_BASELINE = False  # Week 1: never escalate — 0% catch rate is the correct baseline

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

DATA_DIR = Path(__file__).parent.parent / "data" / "products"


# ---------------------------------------------------------------------------
# Context loading — in-memory, no pgvector in Week 1
# ---------------------------------------------------------------------------

def load_product_docs() -> str:
    """Read all *.md files from data/products/ and return as a combined context string."""
    parts = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        parts.append(f"## {md_file.stem}\n\n{md_file.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(parts)


# Load once at import time — avoids re-reading disk on every query
PRODUCT_CONTEXT = load_product_docs()


# ---------------------------------------------------------------------------
# Step 1: Intent Classification
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are an intent classifier for Novus Bank's customer support system.

Classify the customer query into EXACTLY ONE of these intents:
- return_or_refund: prepayment, foreclosure, refunds, reversals, cancellations, disputed charges
- order_status: application status, processing times, account activation, disbursement timelines
- billing_or_payment: EMIs, fees, charges, ATM withdrawals, billing disputes, unauthorised debits
- product_info: product features, interest rates, loan eligibility, limits, debit card features
- membership: tier benefits (Standard, Plus, Elite), upgrades, tier-specific policies
- general: account opening, dormancy, online banking, complaints not fitting above categories

Respond with ONLY the intent name, no explanation, no punctuation."""


def classify_intent(query: str) -> str:
    """Classify query into one of 6 intents using GPT-4o-mini at temperature=0."""
    response = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    intent = response.choices[0].message.content.strip().lower()
    return intent if intent in INTENTS else "general"


# ---------------------------------------------------------------------------
# Step 2: Routing
# ---------------------------------------------------------------------------

def route(intent: str, query: str) -> bool:
    """Decide whether to escalate to a human agent.

    Week 1 intentional baseline: always False (handle automatically).
    Routing accuracy on should-escalate queries = 0% by design.
    See debt/pb-no-escalation-logic — Week 4 will replace this with a
    real escalation classifier.
    """
    return ESCALATION_BASELINE


# ---------------------------------------------------------------------------
# Step 3: Answer Generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT_PREFIX = """You are a helpful customer support agent for Novus Bank.

Answer the customer question using ONLY the information in the product knowledge below.
If the answer is not covered, say: "I don't have specific information about that. \
Please contact our support team."
Be concise, accurate, and professional.

Product Knowledge:
"""


def generate_answer(query: str, context: str) -> str:
    """Generate a grounded answer using product docs as context.

    Context is concatenated via string addition (not .format()) to avoid KeyError
    if any product doc contains literal brace characters.
    """
    system_content = ANSWER_SYSTEM_PROMPT_PREFIX + context
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_query(query: str) -> dict:
    """Process a customer query through classify → route → generate.

    Returns:
        {
            "query": str,
            "intent": str,        one of 6 intent classes
            "escalation": bool,   True = route to human (always False in Week 1)
            "answer": str,        generated response grounded in product docs
            "context": str,       full product docs used as context
            "trace_id": None,     LangFuse trace id (stub in Week 1)
        }
    """
    intent = classify_intent(query)
    escalation = route(intent, query)
    answer = generate_answer(query, PRODUCT_CONTEXT)

    return {
        "query": query,
        "intent": intent,
        "escalation": escalation,
        "answer": answer,
        "context": PRODUCT_CONTEXT,
        "trace_id": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Novus Support Agent pipeline")
    parser.add_argument("--query", type=str, help="Single query to handle")
    args = parser.parse_args()

    queries = (
        [args.query]
        if args.query
        else [
            "What is the minimum balance for a savings account?",
            "Can I prepay my personal loan early?",
            "What are the benefits of Elite membership?",
        ]
    )

    for q in queries:
        print(f"\nQuery   : {q}")
        t0 = time.time()
        result = handle_query(q)
        elapsed = round(time.time() - t0, 2)
        print(f"Intent  : {result['intent']}")
        print(f"Escalate: {result['escalation']}")
        print(f"Answer  : {result['answer']}")
        print(f"Time    : {elapsed}s")
        print("-" * 60)


if __name__ == "__main__":
    main()
