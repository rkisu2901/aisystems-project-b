"""
eval_harness.py — 4-dimensional evaluation harness for the Novus Support Agent.

Dimensions:
  1. Classification accuracy — predicted_intent == expected_intent        (B1.1)
  2. Routing accuracy        — predicted_escalation == expected_escalation (B1.2)
  3. Faithfulness            — LLM judge: answer grounded in product docs? (B1.3)
  4. Correctness             — LLM judge: answer matches expected answer?  (B1.3)

Routing is reported as two sub-metrics:
  - correct-handle rate   : expected_escalation=False, pipeline correctly handles
  - missed-escalation rate: expected_escalation=True,  pipeline correctly catches

Week 1 baseline: correct-handle ≈ 100%, missed-escalation = 0% (pipeline never escalates).

Usage:
    python scripts/eval_harness.py                   # full eval (32 queries)
    python scripts/eval_harness.py --save-baseline   # also saves baseline_scores.json
    python scripts/eval_harness.py --limit 5         # quick smoke test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Windows console defaults to cp1252 which can't print ₹ — force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.support_pipeline import handle_query

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

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
BASELINE_PATH = Path(__file__).parent / "baseline_scores.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Dimension 1: Classification Accuracy  (B1.1)
# ---------------------------------------------------------------------------

def check_classification(predicted_intent: str, expected_intent: str) -> bool:
    """Return True if predicted_intent matches expected_intent exactly.

    Simple string equality — run across all entries, it reveals which intent
    classes the classifier confuses most consistently.
    """
    return predicted_intent == expected_intent


# ---------------------------------------------------------------------------
# Dimension 2: Routing Accuracy  (B1.2)
# ---------------------------------------------------------------------------

def check_routing(predicted_escalation: bool, expected_escalation: bool) -> bool:
    """Return True if escalation decision matches expected.

    Week 1 baseline: pipeline always returns False, so this returns False for
    every entry where expected_escalation=True. That 0% catch rate is the
    correct documented baseline — see debt/pb-no-escalation-logic.
    Week 4 will replace route() with a real escalation classifier.
    """
    return predicted_escalation == expected_escalation


# ---------------------------------------------------------------------------
# Dimension 3: Faithfulness (LLM-as-judge)  (B1.3)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are an evaluation judge for a banking AI assistant.

Score the following answer on FAITHFULNESS: does the answer stay grounded in
the provided context, or does it add information not present in the context?

Scoring rubric:
  5 — Every claim in the answer is directly supported by the context.
  4 — Mostly grounded; minor paraphrasing that doesn't change meaning.
  3 — Partly grounded; some claims go slightly beyond the context.
  2 — Significant information added beyond what the context states.
  1 — Answer largely fabricated or contradicts the context.

Context:
{context}

Answer:
{answer}

Respond with valid JSON only, no other text:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""


def judge_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """Score answer faithfulness relative to product doc context (1-5)."""
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Dimension 4: Correctness (LLM-as-judge)  (B1.3)
# ---------------------------------------------------------------------------

CORRECTNESS_PROMPT = """You are an evaluation judge for a banking AI assistant.

Score the following answer on CORRECTNESS: does it accurately and completely
address the customer's question, compared to the expected answer?

Scoring rubric:
  5 — Fully correct; all key facts match the expected answer.
  4 — Mostly correct; minor omission or slightly different phrasing.
  3 — Partially correct; gets the main point but misses important details.
  2 — Mostly wrong; addresses the question but with significant factual errors.
  1 — Completely wrong or doesn't address the question.

Question: {query}

Expected Answer: {expected_answer}

Actual Answer: {answer}

Respond with valid JSON only, no other text:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""


def judge_correctness(query: str, answer: str, expected_answer: str) -> dict[str, Any]:
    """Score answer correctness against the expected answer (1-5)."""
    prompt = CORRECTNESS_PROMPT.format(
        query=query, expected_answer=expected_answer, answer=answer
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Optional: LangFuse score attachment
# ---------------------------------------------------------------------------

def attach_langfuse_scores(
    trace_id: str,
    faithfulness: float,
    correctness: float,
    classification: float,
    routing: float,
) -> None:
    """Post all 4 dimension scores to a LangFuse trace."""
    if not LANGFUSE_ENABLED or not _lf or not trace_id:
        return
    try:
        for name, value in [
            ("faithfulness", faithfulness),
            ("correctness", correctness),
            ("classification", classification),
            ("routing", routing),
        ]:
            _lf.score(trace_id=trace_id, name=name, value=value)
    except Exception:
        pass  # LangFuse is observability-only; never block the eval loop


# ---------------------------------------------------------------------------
# Main eval loop  (B1.4)
# ---------------------------------------------------------------------------

def run_eval(dataset: list[dict], verbose: bool = True) -> list[dict]:
    """Run the full 4-dimensional eval pipeline over all dataset entries.

    Returns a list of per-entry result dicts with all 4 dimension scores.
    """
    results = []

    for i, entry in enumerate(dataset):
        if verbose:
            print(f"  [{i+1}/{len(dataset)}] {entry['id']}: {entry['query'][:60]}…")

        pipeline_result = handle_query(entry["query"])

        # Dimensions 1 & 2: binary classification and routing checks
        classification_hit = check_classification(
            pipeline_result["intent"], entry["expected_intent"]
        )
        routing_hit = check_routing(
            pipeline_result["escalation"], entry["expected_escalation"]
        )

        # Dimensions 3 & 4: LLM-as-judge
        faith_result = judge_faithfulness(
            pipeline_result["answer"], pipeline_result["context"]
        )
        correct_result = judge_correctness(
            entry["query"], pipeline_result["answer"], entry["expected_answer"]
        )

        if pipeline_result.get("trace_id"):
            attach_langfuse_scores(
                pipeline_result["trace_id"],
                faith_result["score"],
                correct_result["score"],
                int(classification_hit),
                int(routing_hit),
            )

        result = {
            "id": entry["id"],
            "query": entry["query"],
            "category": entry.get("category", "unknown"),
            "difficulty": entry.get("difficulty", "unknown"),
            "expected_intent": entry["expected_intent"],
            "predicted_intent": pipeline_result["intent"],
            "expected_escalation": entry["expected_escalation"],
            "predicted_escalation": pipeline_result["escalation"],
            "expected_answer": entry["expected_answer"],
            "answer": pipeline_result["answer"],
            "classification_hit": classification_hit,
            "routing_hit": routing_hit,
            "faithfulness": faith_result["score"],
            "faithfulness_reason": faith_result["reason"],
            "correctness": correct_result["score"],
            "correctness_reason": correct_result["reason"],
            "trace_id": pipeline_result.get("trace_id"),
        }
        results.append(result)

    return results


def compute_scorecard(results: list[dict]) -> dict:
    """Aggregate per-entry results into a summary scorecard."""
    n = len(results)
    if n == 0:
        return {}

    classification_acc = sum(r["classification_hit"] for r in results) / n
    routing_acc = sum(r["routing_hit"] for r in results) / n

    # Routing breakdown: separate handle-correctly vs escalation-caught rates
    handle_cases = [r for r in results if not r["expected_escalation"]]
    escalate_cases = [r for r in results if r["expected_escalation"]]
    correct_handle_rate = (
        sum(r["routing_hit"] for r in handle_cases) / len(handle_cases)
        if handle_cases else 0.0
    )
    missed_escalation_rate = (
        sum(r["routing_hit"] for r in escalate_cases) / len(escalate_cases)
        if escalate_cases else 0.0
    )

    # Normalize 1–5 judge scores to 0–1 for consistent comparison
    faithfulness_norm = sum(r["faithfulness"] for r in results) / (n * 5)
    correctness_norm = sum(r["correctness"] for r in results) / (n * 5)

    return {
        "n": n,
        "classification_accuracy": round(classification_acc, 4),
        "routing_accuracy": round(routing_acc, 4),
        "correct_handle_rate": round(correct_handle_rate, 4),
        "missed_escalation_rate": round(missed_escalation_rate, 4),
        "n_handle_cases": len(handle_cases),
        "n_escalation_cases": len(escalate_cases),
        "faithfulness": round(faithfulness_norm, 4),
        "correctness": round(correctness_norm, 4),
        "faithfulness_raw": round(sum(r["faithfulness"] for r in results) / n, 2),
        "correctness_raw": round(sum(r["correctness"] for r in results) / n, 2),
    }


# ---------------------------------------------------------------------------
# Stratified eval by intent  (B2.2 stretch)
# ---------------------------------------------------------------------------

def run_stratified_eval(results: list[dict]) -> dict:
    """Per-intent breakdown of all 4 dimensions.

    Groups results by expected_intent. Shows which intent classes the
    classifier gets wrong most, and which have the lowest generation scores.
    """
    by_intent: dict[str, list] = {}
    for r in results:
        by_intent.setdefault(r["expected_intent"], []).append(r)

    intent_breakdown = {}
    for intent, rows in by_intent.items():
        total = len(rows)
        correct_class = sum(r["classification_hit"] for r in rows)
        intent_breakdown[intent] = {
            "total": total,
            "classification_correct": correct_class,
            "classification_accuracy": round(correct_class / total, 4) if total else 0.0,
            "faithfulness_raw": round(sum(r["faithfulness"] for r in rows) / total, 2),
            "correctness_raw": round(sum(r["correctness"] for r in rows) / total, 2),
        }

    return {"by_intent": intent_breakdown}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_scorecard(scorecard: dict, title: str = "Overall Scorecard") -> None:
    width = 58
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Queries evaluated      : {scorecard.get('n', 0)}")
    print(f"  Classification Acc     : {scorecard.get('classification_accuracy', 0):.1%}")
    print(f"  Routing Accuracy       : {scorecard.get('routing_accuracy', 0):.1%}")
    print(f"    ├─ Correct-Handle    : {scorecard.get('correct_handle_rate', 0):.1%}  "
          f"(n={scorecard.get('n_handle_cases', 0)})")
    print(f"    └─ Missed-Escalation : {scorecard.get('missed_escalation_rate', 0):.1%}  "
          f"(n={scorecard.get('n_escalation_cases', 0)})  ← Week 4 target")
    print(f"  Faithfulness           : {scorecard.get('faithfulness_raw', 0):.2f}/5.0  "
          f"({scorecard.get('faithfulness', 0):.1%})")
    print(f"  Correctness            : {scorecard.get('correctness_raw', 0):.2f}/5.0  "
          f"({scorecard.get('correctness', 0):.1%})")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Novus Support Agent eval harness")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save aggregate scores as baseline_scores.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N entries (smoke test)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-entry progress output")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"\nRunning eval on {len(dataset)} queries…\n")
    t0 = time.time()
    results = run_eval(dataset, verbose=not args.quiet)
    elapsed = round(time.time() - t0, 1)

    scorecard = compute_scorecard(results)
    stratified = run_stratified_eval(results)

    print_scorecard(scorecard, title="Novus Support Agent — Overall Scorecard")

    # Per-intent breakdown (stretch B2.2)
    print("Per-intent classification breakdown:")
    for intent, stats in sorted(
        stratified["by_intent"].items(),
        key=lambda kv: kv[1]["classification_accuracy"],
    ):
        print(
            f"  {intent:<25}  n={stats['total']:>2}  "
            f"class={stats['classification_accuracy']:.0%}  "
            f"faith={stats['faithfulness_raw']:.1f}  "
            f"correct={stats['correctness_raw']:.1f}"
        )

    print(f"\nTotal eval time: {elapsed}s")

    output = {
        "scorecard": scorecard,
        "stratified": stratified,
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results saved → {RESULTS_PATH}")

    if args.save_baseline:
        baseline = {"scorecard": scorecard, "stratified": stratified}
        BASELINE_PATH.write_text(
            json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Baseline saved → {BASELINE_PATH}")


if __name__ == "__main__":
    main()
