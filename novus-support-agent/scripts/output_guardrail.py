"""
output_guardrail.py — Output safety guardrail for the Novus Support Agent.

check_hallucination() verifies every factual claim in a generated answer
against the retrieved context. If any claim is ungrounded it flags the answer
before it reaches the user.

Returns:
    {
        "has_hallucination": bool,
        "claims":            list[dict],   all claims with supported + evidence
        "unsupported_claims": list[str],   text of claims not in context
    }

Usage (import):
    from scripts.output_guardrail import check_hallucination
    result = check_hallucination(answer, context)

Usage (CLI — runs O2.1 test suite: 3 grounded + 3 hallucinated):
    python scripts/output_guardrail.py
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL       = "gpt-4o-mini"
JUDGE_TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Judge prompt (O2.1)
# ---------------------------------------------------------------------------

HALLUCINATION_PROMPT = """You are a fact-checking judge.
Given a context and an answer, identify every factual claim in the answer.
For each claim, state whether it is SUPPORTED or UNSUPPORTED by the context.

Context:
{context}

Answer:
{answer}

Rules:
- A claim is SUPPORTED only if the context explicitly states it. Inferences and paraphrases that change specific values count as UNSUPPORTED.
- A claim is UNSUPPORTED if the context is silent on it or contradicts it.
- Respond with JSON only. No markdown fences. No explanation outside the JSON.

Response format:
{{"claims": [{{"claim": "<str>", "supported": <bool>, "evidence": "<str or empty>"}}], "has_hallucination": <bool>}}"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def check_hallucination(answer: str, context: str) -> dict:
    """Verify every factual claim in answer against context.

    Returns:
        {
            "has_hallucination":  bool,
            "claims":             list[dict],  each: {claim, supported, evidence}
            "unsupported_claims": list[str],   claim text for unsupported entries
        }
    """
    prompt = HALLUCINATION_PROMPT.format(
        context=context[:4000],  # guard against very large context windows
        answer=answer,
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        # Fail closed: API error → treat as hallucination detected
        return {
            "has_hallucination":  True,
            "claims":             [],
            "unsupported_claims": ["[api_error — could not verify claims]"],
        }

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "has_hallucination":  True,
            "claims":             [],
            "unsupported_claims": ["[parse_error — could not verify claims]"],
        }

    claims = parsed.get("claims", [])
    # Defensive: re-derive has_hallucination from claims in case model is inconsistent
    any_unsupported     = any(not c.get("supported", True) for c in claims)
    has_hallucination   = any_unsupported or bool(parsed.get("has_hallucination", False))
    unsupported_claims  = [c["claim"] for c in claims if not c.get("supported", True)]

    return {
        "has_hallucination":  has_hallucination,
        "claims":             claims,
        "unsupported_claims": unsupported_claims,
    }


# ---------------------------------------------------------------------------
# O2.1 test suite
# ---------------------------------------------------------------------------

SAVINGS_CONTEXT = """
Minimum balance: ₹1,000 (Standard), waived for Plus/Elite.
Interest rate: 3.5% p.a. (Standard), 4.0% (Plus/Elite), credited quarterly.
ATM withdrawals: 3 free/month at other banks (Standard); 5 free (Plus); unlimited (Elite).
Debit card: Included free; contactless up to ₹5,000 without PIN.
Online banking: Full app + net banking access.
Account opening: Video KYC or branch; activation within 2 hours.
Dormancy: Account becomes dormant after 24 months of inactivity.
""".strip()

LOAN_CONTEXT = """
Eligibility: CIBIL score ≥ 750; salaried or self-employed.
Loan amount: ₹50,000 – ₹25,00,000.
Tenure: 12 – 60 months.
Interest rate: 11% (Elite), 13% (Plus), 16–18% (Standard) p.a.
Processing fee: 1% of loan amount (min ₹999).
Disbursement: Within 24 hours for pre-approved customers; 3–5 working days otherwise.
Prepayment: Allowed after 6 EMIs; 2% penalty on outstanding amount.
Foreclosure: After 12 EMIs; 1% charge. Elite customers: zero charge after 18 EMIs.
EMI bounce fee: ₹500 per instance.
Late interest: 2% per month on overdue amount.
""".strip()

TEST_CASES = [
    # --- GROUNDED (no hallucination expected) ---
    {
        "id": "G1",
        "label": "GROUNDED",
        "context": SAVINGS_CONTEXT,
        "answer": (
            "The minimum balance for a Standard savings account is ₹1,000. "
            "Plus and Elite accounts have no minimum balance requirement. "
            "Interest is credited quarterly."
        ),
    },
    {
        "id": "G2",
        "label": "GROUNDED",
        "context": LOAN_CONTEXT,
        "answer": (
            "Personal loan processing fee is 1% of the loan amount, "
            "with a minimum of ₹999. Disbursement for pre-approved customers "
            "happens within 24 hours."
        ),
    },
    {
        "id": "G3",
        "label": "GROUNDED",
        "context": LOAN_CONTEXT,
        "answer": (
            "Prepayment is allowed after completing 6 EMIs. "
            "A penalty of 2% on the outstanding amount applies. "
            "Elite customers can foreclose with zero charge after 18 EMIs."
        ),
    },
    # --- HALLUCINATED (hallucination expected) ---
    {
        "id": "H1",
        "label": "HALLUCINATED",
        "context": SAVINGS_CONTEXT,
        "answer": (
            "Elite savings account holders earn 5.0% interest per annum, "
            "credited monthly to the account."
        ),
        # Truth: rate is 4.0% for Elite, credited quarterly — not 5.0% or monthly
    },
    {
        "id": "H2",
        "label": "HALLUCINATED",
        "context": LOAN_CONTEXT,
        "answer": (
            "Elite customers pay zero EMI bounce fee. "
            "The standard EMI bounce fee is ₹500 but this is waived for Elite members."
        ),
        # Truth: ₹500 bounce fee applies to all; no Elite waiver in corpus
    },
    {
        "id": "H3",
        "label": "HALLUCINATED",
        "context": LOAN_CONTEXT,
        "answer": (
            "The personal loan processing fee is waived for pre-approved Elite customers. "
            "Disbursement for these customers happens within 12 hours."
        ),
        # Truth: processing fee is 1% for all (no Elite waiver); disbursement is 24h, not 12h
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== O2.1 — Hallucination detector test (3 grounded + 3 hallucinated) ===\n")

    grounded_correct    = 0
    hallucinated_caught = 0

    for tc in TEST_CASES:
        result = check_hallucination(tc["answer"], tc["context"])
        detected = result["has_hallucination"]
        expected = tc["label"] == "HALLUCINATED"

        correct = detected == expected
        if tc["label"] == "GROUNDED" and not detected:
            grounded_correct += 1
        if tc["label"] == "HALLUCINATED" and detected:
            hallucinated_caught += 1

        verdict = "✓" if correct else "✗ WRONG"
        print(f"[{tc['id']}] {tc['label']:<12}  hallucination={str(detected):<5}  {verdict}")
        print(f"  Answer: {tc['answer'][:90]}...")
        if result["unsupported_claims"]:
            for uc in result["unsupported_claims"]:
                print(f"  ⚠ Unsupported: {uc}")
        print(f"  Claims checked: {len(result['claims'])}")
        print()

    print(f"Grounded correctly passed : {grounded_correct}/3")
    print(f"Hallucinated correctly caught: {hallucinated_caught}/3")


if __name__ == "__main__":
    main()
