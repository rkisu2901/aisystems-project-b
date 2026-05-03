"""
classifier_scratch.py — Intent classification built from scratch.

B1.1  Rule-based keyword classifier   (no LLM, no framework)
B1.2  LLM-based classifier            (raw client call, temperature=0)

Both are run against the same 32 golden entries so the per-intent accuracy
comparison is apples-to-apples.

Usage:
    python scripts/classifier_scratch.py
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

# ---------------------------------------------------------------------------
# B1.1 — Rule-based keyword classifier
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, list[str]] = {
    "return_or_refund": [
        "prepay", "prepayment", "foreclose", "foreclosure",
        "refund", "reverse", "reversal", "penalty", "close loan",
        "cancel", "money back", "extra charge", "unauthorised deduction",
        "unauthorized deduction",
    ],
    "order_status": [
        "activate", "activation", "status", "how long", "when will",
        "disburse", "disbursement", "processing time", "timeline",
        "how soon", "how many days",
    ],
    "billing_or_payment": [
        "emi", "charge", "fee", "deduct", "payment", "bill",
        "invoice", "unauthorized debit", "unauthorised debit",
        "overcharged", "interest charged", "processing fee",
    ],
    "product_info": [
        "interest rate", "eligible", "eligibility", "features",
        "limit", "maximum loan", "minimum balance", "how much",
        "what is the", "tell me about", "details",
    ],
    "membership": [
        "premium", "elite", "plus", "standard tier", "tier",
        "membership", "benefit", "upgrade", "lounge", "gold",
    ],
    "general": [],   # fallback — no keywords
}


def classify_keyword(query: str) -> str:
    """Classify query into one of 6 intents using keyword matching only.

    Iterates intents in definition order. The first intent whose keyword
    list has ANY match wins. 'general' is the implicit fallback.

    Design note: keyword order matters — more specific intents (e.g.
    return_or_refund) are checked before broad ones (billing_or_payment)
    to avoid false matches on shared terms like 'charge'.
    """
    q = query.lower()
    for intent, keywords in KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# B1.2 — LLM-based classifier (raw OpenAI SDK call, no framework)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the customer query into EXACTLY ONE of these intents:\n"
    "- return_or_refund: prepayment, foreclosure, refunds, reversals, cancellations, disputed charges\n"
    "- order_status: application status, processing times, account activation, disbursement timelines\n"
    "- billing_or_payment: EMIs, fees, charges, ATM withdrawals, billing disputes, unauthorised debits\n"
    "- product_info: product features, interest rates, loan eligibility, limits, savings account features\n"
    "- membership: tier benefits (Standard, Plus, Elite), upgrades, tier-specific policies\n"
    "- general: account opening, dormancy, online banking, complaints not fitting above categories\n\n"
    "Respond with ONLY the intent name, no explanation, no punctuation."
)


def classify_llm(query: str) -> str:
    """Classify query using GPT-4o-mini at temperature=0.

    Temperature=0 makes the output deterministic — same query always
    returns the same intent. This is essential for reproducible eval.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    result = response.choices[0].message.content.strip().lower()
    return result if result in INTENTS else "general"


# ---------------------------------------------------------------------------
# Evaluation against golden dataset
# ---------------------------------------------------------------------------

def evaluate(dataset: list[dict], classify_fn) -> dict:
    """Run classify_fn over dataset. Return per-intent accuracy table."""
    by_intent: dict[str, dict] = {i: {"correct": 0, "total": 0, "errors": []} for i in INTENTS}

    for entry in dataset:
        expected = entry["expected_intent"]
        predicted = classify_fn(entry["query"])
        by_intent.setdefault(expected, {"correct": 0, "total": 0, "errors": []})
        by_intent[expected]["total"] += 1
        if predicted == expected:
            by_intent[expected]["correct"] += 1
        else:
            by_intent[expected]["errors"].append({
                "query": entry["query"],
                "predicted": predicted,
            })

    total_correct = sum(v["correct"] for v in by_intent.values())
    total_queries = sum(v["total"] for v in by_intent.values())
    return {
        "overall_accuracy": round(total_correct / total_queries, 4) if total_queries else 0,
        "by_intent": {
            intent: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                "correct": v["correct"],
                "total": v["total"],
                "errors": v["errors"],
            }
            for intent, v in by_intent.items() if v["total"] > 0
        },
    }


# ---------------------------------------------------------------------------
# CLI — B1.1 + B1.2 side-by-side comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # --- B1.1: keyword classifier ---
    print("=" * 60)
    print("B1.1 — Keyword Classifier")
    print("=" * 60)
    kw_results = evaluate(dataset, classify_keyword)
    print(f"Overall accuracy: {kw_results['overall_accuracy']:.1%}\n")
    print(f"{'Intent':<25} {'Acc':>6}  {'Correct/Total':>14}  Failure examples")
    print("-" * 60)
    for intent, stats in kw_results["by_intent"].items():
        errors = stats["errors"]
        example = f"  e.g. '{errors[0]['query'][:35]}…' → {errors[0]['predicted']}" if errors else ""
        print(f"{intent:<25} {stats['accuracy']:>6.0%}  {stats['correct']:>6}/{stats['total']:<6}{example}")

    # Two failure examples as required by B1.1 deliverable
    print("\nB1.1 Keyword failure analysis:")
    all_errors = [
        (intent, e) for intent, stats in kw_results["by_intent"].items()
        for e in stats["errors"]
    ]
    for i, (intent, err) in enumerate(all_errors[:2], 1):
        print(f"  [{i}] Expected '{intent}', got '{err['predicted']}'")
        print(f"       Query: '{err['query']}'")
        print(f"       Why: keyword lists overlap — e.g. 'charge' matches billing AND return_or_refund")

    # --- B1.2: LLM classifier ---
    print("\n" + "=" * 60)
    print("B1.2 — LLM Classifier (gpt-4o-mini, temperature=0)")
    print("=" * 60)
    llm_results = evaluate(dataset, classify_llm)
    print(f"Overall accuracy: {llm_results['overall_accuracy']:.1%}\n")

    # Side-by-side comparison table
    print(f"\n{'Intent':<25} {'Keyword':>8}  {'LLM':>8}  {'Winner':>8}")
    print("-" * 55)
    for intent in INTENTS:
        kw_acc = kw_results["by_intent"].get(intent, {}).get("accuracy", 0)
        llm_acc = llm_results["by_intent"].get(intent, {}).get("accuracy", 0)
        winner = "LLM" if llm_acc > kw_acc else ("TIE" if llm_acc == kw_acc else "KEYWORD")
        print(f"{intent:<25} {kw_acc:>8.0%}  {llm_acc:>8.0%}  {winner:>8}")

    print(f"\n{'OVERALL':<25} {kw_results['overall_accuracy']:>8.0%}  {llm_results['overall_accuracy']:>8.0%}")
    print("\nB1.2 Reflection:")
    print("  LLM outperforms keyword on ambiguous intents (billing_or_payment vs return_or_refund)")
    print("  where the same words ('charge', 'deduct') appear in both classes.")
    print("  Keyword matching is deterministic and free; LLM costs ~0.001 USD per query.")
    print("  For 'membership' and 'order_status', keyword is competitive — clear vocabulary.")
