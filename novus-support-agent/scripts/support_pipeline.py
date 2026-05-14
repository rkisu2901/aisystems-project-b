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

# ---------------------------------------------------------------------------
# Week 4 — Guardrails (opt-in)
# ---------------------------------------------------------------------------

try:
    from scripts.input_guardrail import check_input as _check_input
    INPUT_GUARDRAIL_AVAILABLE = True
except ImportError:
    INPUT_GUARDRAIL_AVAILABLE = False

try:
    from scripts.output_guardrail import check_hallucination as _check_hallucination
    OUTPUT_GUARDRAIL_AVAILABLE = True
except ImportError:
    OUTPUT_GUARDRAIL_AVAILABLE = False

FALLBACK_ANSWER = (
    "I found some relevant information but cannot confirm all the details with "
    "full accuracy. Please contact Novus Bank support at 1800-NOVUS for a "
    "verified answer to your question."
)

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
# Step 2: Routing (escalation decision — Week 4 real classifier)
# ---------------------------------------------------------------------------

_ESCALATION_PROMPT = """You are an escalation classifier for Novus Bank customer support.

Decide whether this query must be escalated to a human agent or can be handled automatically.

Escalate (respond "YES") when ANY of the following are true:
- The customer reports fraud, unauthorized transaction, or account compromise
- The customer reports SIM swap or suspicious login activity
- The customer is distressed, threatening legal action, or using urgent/angry language
- The query involves a disputed charge above ₹10,000
- The customer explicitly requests to speak to a human agent
- The situation involves a potential regulatory/compliance concern

Handle automatically (respond "NO") for:
- Standard product information queries
- Policy questions (rates, limits, fees)
- Account activation, KYC, or onboarding questions
- General how-to questions

Respond with exactly one word: YES or NO."""


def route(intent: str, query: str) -> bool:
    """Escalation routing — LLM-based classifier (Week 4).

    Returns True if the query should be escalated to a human agent.
    Fails safe (returns False) on any API error so the pipeline
    can still serve an automated answer rather than dropping the query.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": _ESCALATION_PROMPT},
                {"role": "user",   "content": f"Intent: {intent}\nQuery: {query}"},
            ],
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False  # fail safe: serve automated answer on API error


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

_STRICT_SYSTEM_PROMPT_PREFIX = """You are a strict customer support agent for Novus Bank.

CRITICAL: Answer ONLY from the product knowledge provided below.
- If a specific number, rate, or policy detail is not explicitly stated, do NOT include it.
- Quote or closely paraphrase the source. Do not infer, extrapolate, or add details.
- If the answer is not present, respond: "I don't have specific information about that. Please contact our support team at 1800-NOVUS."

Product Knowledge:
"""


def generate_answer(query: str, context: str, _strict: bool = False) -> str:
    """Generate a grounded answer using LiteLLM with automatic fallback.

    Uses litellm.completion() if LiteLLM is installed — falls back to
    gpt-3.5-turbo automatically if gpt-4o-mini fails (quota, outage).
    Falls back to raw OpenAI SDK if LiteLLM is not installed.

    Context is concatenated via string addition (not .format()) to be
    safe against literal brace characters in product doc content.
    _strict=True uses a stricter system prompt that forbids inference — used
    on the retry pass after the output guardrail detects a hallucination.
    """
    prefix = _STRICT_SYSTEM_PROMPT_PREFIX if _strict else ANSWER_SYSTEM_PROMPT_PREFIX
    system_content = prefix + context
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

def handle_query(
    query: str,
    use_anonymizer: bool = False,
    use_guardrail: bool = False,
    use_output_guardrail: bool = False,
    guardrail_sample_rate: float = 1.0,
) -> dict:
    """Process a customer query through classify → route → retrieve → generate.

    Week 2 changes vs Week 1:
      - intent now returned alongside tool (routing is visible in result)
      - context is filtered to intent-relevant docs and deduplicated
      - answer generated via LiteLLM with fallback

    Week 4 additions:
      - use_anonymizer=True (P3.2): PII stripped before any LLM call, restored in answer.
      - use_guardrail=True: check_input() fires first; blocked queries return immediately.
      - use_output_guardrail=True: check_hallucination() runs after generation; retries once
        with _STRICT_SYSTEM_PROMPT on detection; returns FALLBACK_ANSWER on double failure.
      - guardrail_sample_rate: fraction of queries checked by output guardrail (0.0–1.0).
      All flags default to False — eval harness unaffected.

    Returns:
        {
            "query":               str,
            "intent":              str,
            "tool":                str,
            "escalation":          bool,
            "answer":              str,
            "context":             str,
            "pii_redacted":        bool,
            "trace_id":            None,
            "guardrail_blocked":   bool,
            "guardrail_reason":    str | None,
            "hallucination_detected": bool | None,
        }
    """
    import random
    t0 = time.time()

    # --- Input guardrail (G1.1/G1.2) — fires before PII/classification/retrieval ---
    if use_guardrail and INPUT_GUARDRAIL_AVAILABLE:
        guard = _check_input(query)
        if not guard["safe"]:
            return {
                "query":               query,
                "intent":              "blocked",
                "tool":                "input_guardrail",
                "escalation":          False,
                "answer":              guard["refusal"],
                "context":             "",
                "pii_redacted":        False,
                "trace_id":            None,
                "guardrail_blocked":   True,
                "guardrail_reason":    guard["category"],
                "hallucination_detected": None,
            }

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

    # --- Output guardrail (O2.1) — verify answer against context, with retry ---
    hallucination_detected = None
    if (
        use_output_guardrail
        and OUTPUT_GUARDRAIL_AVAILABLE
        and random.random() < guardrail_sample_rate
    ):
        og_result = _check_hallucination(raw_answer, context)
        if og_result["has_hallucination"]:
            # Retry once with strict system prompt
            raw_answer = generate_answer(clean_query, context, _strict=True)
            og_result2 = _check_hallucination(raw_answer, context)
            if og_result2["has_hallucination"]:
                raw_answer = FALLBACK_ANSWER
            hallucination_detected = True
        else:
            hallucination_detected = False

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
        "query":               query,        # original — do NOT log if pii_redacted=True
        "intent":              intent,
        "tool":                tool,
        "escalation":          escalation,
        "answer":              answer,
        "context":             context,
        "pii_redacted":        pii_redacted,
        "trace_id":            None,
        "guardrail_blocked":   False,
        "guardrail_reason":    None,
        "hallucination_detected": hallucination_detected,
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
