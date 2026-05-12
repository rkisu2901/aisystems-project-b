"""
support_pipeline.py — Novus Support Agent query-handling pipeline.

Week 2 upgrades over Week 1:
  - classify_intent() now uses query_classifier.classify_tool() for tool routing
  - Context loaded via retrieval.retrieve_filtered() (intent-aware, deduplicated)
  - generate_answer() uses LiteLLM with gpt-3.5-turbo fallback (B2.3 stretch)

Week 4 additions:
  - PiiAnonymizer opt-in (use_anonymizer=True): anonymize → pipeline → restore (P3.2)

Pipeline with anonymizer enabled:
  1. PiiAnonymizer.anonymize(query)   — replace PII with typed placeholders
  2. classify_tool(clean_query)       — intent + tool routing on anonymized text
  3. route()                          — escalation decision
  4. load_context(intent)             — filtered + deduplicated product docs
  5. generate_answer(clean_query)     — LiteLLM grounded answer (sees no raw PII)
  6. PiiAnonymizer.restore(answer)    — replace placeholders with original values

Public entry point:
    handle_query(query: str, use_anonymizer: bool = False) -> dict

Usage:
    python scripts/support_pipeline.py                   # 3 built-in test queries
    python scripts/support_pipeline.py --query "..."     # single query
    python scripts/support_pipeline.py --test            # run all 9 test queries
    python scripts/support_pipeline.py --pii-test        # run 4 PII-containing queries
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# LiteLLM (B2.3 stretch) — model-agnostic with automatic fallback
# ---------------------------------------------------------------------------

try:
    import litellm
    litellm.set_verbose = False
    LITELLM_ENABLED = True
except ImportError:
    LITELLM_ENABLED = False

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

ANSWER_MODEL   = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"
TEMPERATURE    = 0.1
ESCALATION_BASELINE = False   # Week 1 design: always handle, 0% escalation catch rate

# ---------------------------------------------------------------------------
# Week 4 — PII anonymizer (P3.2, opt-in)
# ---------------------------------------------------------------------------

try:
    from scripts.pii_anonymizer import PiiAnonymizer, redaction_audit_log
    PII_ANONYMIZER_AVAILABLE = True
except ImportError:
    PII_ANONYMIZER_AVAILABLE = False

INTENTS = [
    "return_or_refund", "order_status", "billing_or_payment",
    "product_info", "membership", "general",
]

# ---------------------------------------------------------------------------
# Step 1: Intent classification + tool routing (Week 2)
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> tuple[str, str]:
    """Return (intent, tool) using the Week 2 tool router.

    Falls back to direct LLM classification if query_classifier import fails.
    """
    try:
        from scripts.query_classifier import classify_tool
        tool, intent = classify_tool(query)
        return intent, tool
    except Exception:
        # Graceful fallback to direct LLM call (Week 1 behaviour)
        from scripts.classifier_scratch import classify_llm
        intent = classify_llm(query)
        return intent, "policy_kb"


# ---------------------------------------------------------------------------
# Step 2: Routing (escalation decision — Week 1 baseline unchanged)
# ---------------------------------------------------------------------------

def route(intent: str, query: str) -> bool:
    """Escalation routing.

    Week 1 baseline: always return False (handle automatically).
    Week 4 will replace this with a real escalation classifier.
    See debt/pb-no-escalation-logic.
    """
    return ESCALATION_BASELINE


# ---------------------------------------------------------------------------
# Step 3: Context loading via filtered + deduplicated retrieval (Week 2)
# ---------------------------------------------------------------------------

def load_context(intent: str) -> str:
    """Return product doc context filtered to the intent and deduplicated.

    Week 1 loaded all docs unconditionally. Week 2 uses retrieve_filtered()
    so a membership query doesn't receive loan EMI schedules as context,
    reducing hallucination risk and prompt token cost.
    """
    try:
        from scripts.retrieval import retrieve_filtered, deduplicate_chunks
        chunks = retrieve_filtered(intent)
        unique, _ = deduplicate_chunks(chunks, threshold=0.75)
        return "\n\n---\n\n".join(c["content"] for c in unique)
    except Exception:
        # Fallback: read all product docs (Week 1 behaviour)
        data_dir = Path(__file__).parent.parent / "data" / "products"
        parts = []
        for md_file in sorted(data_dir.glob("*.md")):
            parts.append(f"## {md_file.stem}\n\n{md_file.read_text(encoding='utf-8')}")
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 4: Answer generation — LiteLLM with fallback (B2.3)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT_PREFIX = """You are a helpful customer support agent for Novus Bank.

Answer the customer question using ONLY the information in the product knowledge below.
Tier-specific features (Standard, Plus, Elite) may appear inline across multiple entries — \
synthesize them into a coherent answer when relevant.
If the answer is genuinely not present in the product knowledge, say: \
"I don't have specific information about that. Please contact our support team."
Be concise, accurate, and professional.

Product Knowledge:
"""


def generate_answer(query: str, context: str) -> str:
    """Generate a grounded answer using LiteLLM with automatic fallback.

    Uses litellm.completion() if LiteLLM is installed — falls back to
    gpt-3.5-turbo automatically if gpt-4o-mini fails (quota, outage).
    Falls back to raw OpenAI SDK if LiteLLM is not installed.

    Context is concatenated via string addition (not .format()) to be
    safe against literal brace characters in product doc content.
    """
    system_content = ANSWER_SYSTEM_PROMPT_PREFIX + context
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]

    if LITELLM_ENABLED:
        response = litellm.completion(
            model=ANSWER_MODEL,
            fallbacks=[FALLBACK_MODEL],
            messages=messages,
            temperature=TEMPERATURE,
        )
    else:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=TEMPERATURE,
            messages=messages,
        )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_query(query: str, use_anonymizer: bool = False) -> dict:
    """Process a customer query through classify → route → retrieve → generate.

    Week 2 changes vs Week 1:
      - intent now returned alongside tool (routing is visible in result)
      - context is filtered to intent-relevant docs and deduplicated
      - answer generated via LiteLLM with fallback

    Week 4 addition (P3.2):
      - use_anonymizer=True: PII stripped before any LLM call, restored in answer.
        Eval harness leaves this False so results stay deterministic.

    Returns:
        {
            "query":        str,   original query (never log if pii_redacted=True)
            "intent":       str,   one of 6 intent classes
            "tool":         str,   one of policy_kb / order_tracker / account_lookup / multi_tool
            "escalation":   bool,  always False in Week 1/2 baseline
            "answer":       str,   final answer with PII restored (if anonymizer enabled)
            "context":      str,   filtered + deduplicated product docs used
            "pii_redacted": bool,  True if PII was found and anonymized
            "trace_id":     None,
        }
    """
    # --- PII anonymization (P3.2) ---
    # New instance per request; anonymize before ANY LLM call.
    anonymizer  = None
    clean_query = query
    if use_anonymizer and PII_ANONYMIZER_AVAILABLE:
        anonymizer  = PiiAnonymizer()
        clean_query = anonymizer.anonymize(query)

    intent, tool = classify_intent(clean_query)
    escalation   = route(intent, clean_query)
    context      = load_context(intent)
    raw_answer   = generate_answer(clean_query, context)

    # --- Restore PII in the answer ---
    answer = anonymizer.restore(raw_answer) if anonymizer else raw_answer
    pii_redacted = bool(anonymizer and anonymizer.has_pii())

    # --- P3.3: Audit log — only when PII was found ---
    if anonymizer and pii_redacted:
        try:
            redaction_audit_log(
                query=query,
                anonymizer=anonymizer,
                intent=intent,
                trace_id=None,
            )
        except Exception:
            pass  # audit log failure must never break the pipeline

    return {
        "query":        query,        # original — do NOT log if pii_redacted=True
        "intent":       intent,
        "tool":         tool,
        "escalation":   escalation,
        "answer":       answer,
        "context":      context,
        "pii_redacted": pii_redacted,
        "trace_id":     None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


PII_TEST_QUERIES = [
    "My email is priya@gmail.com, order ORD-445521 — where is my refund?",
    "Call me at +91 98765 43210 about the wrong charge on my loan",
    "I'm Rahul Mehta and I was billed twice for ORD-887766",
    "Please check ORD-112233 for test.user@novusbank.com",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Novus Support Agent pipeline")
    parser.add_argument("--query",    type=str, help="Single query to handle")
    parser.add_argument("--test",     action="store_true", help="Run all 9 test queries")
    parser.add_argument("--pii-test", action="store_true", help="Run 4 PII queries with anonymizer")
    args = parser.parse_args()

    if args.pii_test:
        print("=== P3.2 — PII anonymizer wired into handle_query() ===\n")
        for q in PII_TEST_QUERIES:
            t0 = time.time()
            result = handle_query(q, use_anonymizer=True)
            elapsed = round(time.time() - t0, 2)
            print(f"Query        : {q}")
            print(f"pii_redacted : {result['pii_redacted']}")
            print(f"Intent       : {result['intent']}  →  Tool: {result['tool']}")
            print(f"Answer       : {result['answer'][:120]}")
            print(f"Time         : {elapsed}s")
            print("-" * 70)
        return

    if args.test:
        queries = TEST_QUERIES
    elif args.query:
        queries = [args.query]
    else:
        queries = TEST_QUERIES[:3]

    for q in queries:
        print(f"\nQuery   : {q}")
        t0 = time.time()
        result = handle_query(q)
        elapsed = round(time.time() - t0, 2)
        print(f"Intent  : {result['intent']}  →  Tool: {result['tool']}")
        print(f"Escalate: {result['escalation']}")
        print(f"Answer  : {result['answer']}")
        print(f"Time    : {elapsed}s")
        print("-" * 60)


if __name__ == "__main__":
    main()
