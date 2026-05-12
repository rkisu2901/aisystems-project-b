"""
eval_harness.py — 4-dimensional evaluation harness for the Novus Support Agent.

Default mode: LangGraph agent (run_agent from agent.py).
Naive mode:   pass --naive to use the Week 2 handle_query() pipeline instead.

Dimensions:
  1. Classification accuracy — predicted_intent == expected_intent        (B1.1)
  2. Routing accuracy        — predicted_escalation == expected_escalation (B1.2)
  3. Faithfulness            — LLM judge: answer grounded in product docs? (B1.3)
  4. Correctness             — LLM judge: answer matches expected answer?  (B1.3)

Routing is reported as two sub-metrics:
  - correct-handle rate   : expected_escalation=False, pipeline correctly handles
  - missed-escalation rate: expected_escalation=True,  pipeline correctly catches

Agent-only extras (ignored in --naive mode):
  - steps_taken distribution: how many node executions per query (1 / 2 / 3+)
  - tools_called: ordered list of tools for each query

Usage:
    python scripts/eval_harness.py                   # agent mode (default)
    python scripts/eval_harness.py --naive           # naive pipeline (Week 2 baseline)
    python scripts/eval_harness.py --save-baseline   # save baseline_scores.json
    python scripts/eval_harness.py --limit 5         # quick smoke test
    python scripts/eval_harness.py --compare         # run both modes and print delta table
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

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

DATASET_PATH  = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH  = Path(__file__).parent / "eval_results.json"
BASELINE_PATH = Path(__file__).parent / "baseline_scores.json"

client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Dimension 1: Classification Accuracy  (B1.1)
# ---------------------------------------------------------------------------

def check_classification(predicted_intent: str, expected_intent: str) -> bool:
    return predicted_intent == expected_intent


# ---------------------------------------------------------------------------
# Dimension 2: Routing Accuracy  (B1.2)
# ---------------------------------------------------------------------------

def check_routing(predicted_escalation: bool, expected_escalation: bool) -> bool:
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
        pass


# ---------------------------------------------------------------------------
# Main eval loop  (B1.4)
# ---------------------------------------------------------------------------

def run_eval(
    dataset: list[dict],
    verbose: bool = True,
    use_agent: bool = True,
) -> list[dict]:
    """Run the full 4-dimensional eval pipeline over all dataset entries.

    use_agent=True  (default): calls run_agent() from agent.py — LangGraph.
    use_agent=False (--naive): calls handle_query() from support_pipeline.py.

    Agent results include steps_taken and tools_called; naive results have
    those fields set to 0 and [] respectively for schema consistency.
    """
    if use_agent:
        from scripts.agent import run_agent as _call
        def call_pipeline(query: str) -> dict:
            return _call(query)
    else:
        from scripts.support_pipeline import handle_query as _call  # type: ignore[assignment]
        def call_pipeline(query: str) -> dict:
            result = _call(query)
            result.setdefault("steps_taken", 0)
            result.setdefault("tools_called", [])
            return result

    mode_label = "agent" if use_agent else "naive"
    results = []
    for i, entry in enumerate(dataset):
        if verbose:
            print(f"  [{i+1}/{len(dataset)}] [{mode_label}] {entry['id']}: {entry['query'][:55]}…")

        pipeline_result = call_pipeline(entry["query"])

        # Skip LLM judging when the agent itself failed — prevents spurious 5/5
        # faithfulness scores from an empty context string.
        agent_errored = pipeline_result.get("answer", "").startswith("Agent error:")

        classification_hit = check_classification(
            pipeline_result["intent"], entry["expected_intent"]
        )
        routing_hit = check_routing(
            pipeline_result["escalation"], entry["expected_escalation"]
        )

        if agent_errored:
            faith_result   = {"score": 0, "reason": "agent error — skipped"}
            correct_result = {"score": 0, "reason": "agent error — skipped"}
        else:
            faith_result   = judge_faithfulness(pipeline_result["answer"], pipeline_result["context"])
            correct_result = judge_correctness(entry["query"], pipeline_result["answer"], entry["expected_answer"])

        if pipeline_result.get("trace_id"):
            attach_langfuse_scores(
                pipeline_result["trace_id"],
                faith_result["score"],
                correct_result["score"],
                int(classification_hit),
                int(routing_hit),
            )

        result = {
            "id":                    entry["id"],
            "query":                 entry["query"],
            "category":              entry.get("category", "unknown"),
            "difficulty":            entry.get("difficulty", "unknown"),
            "expected_intent":       entry["expected_intent"],
            "predicted_intent":      pipeline_result["intent"],
            "expected_escalation":   entry["expected_escalation"],
            "predicted_escalation":  pipeline_result["escalation"],
            "expected_answer":       entry["expected_answer"],
            "answer":                pipeline_result["answer"],
            "classification_hit":    classification_hit,
            "routing_hit":           routing_hit,
            "faithfulness":          faith_result["score"],
            "faithfulness_reason":   faith_result["reason"],
            "correctness":           correct_result["score"],
            "correctness_reason":    correct_result["reason"],
            "trace_id":              pipeline_result.get("trace_id"),
            # Agent-specific trajectory fields (0 / [] in naive mode)
            "steps_taken":           pipeline_result.get("steps_taken", 0),
            "tools_called":          pipeline_result.get("tools_called", []),
        }
        results.append(result)

    return results


def compute_scorecard(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    classification_acc = sum(r["classification_hit"] for r in results) / n
    routing_acc        = sum(r["routing_hit"]        for r in results) / n

    handle_cases   = [r for r in results if not r["expected_escalation"]]
    escalate_cases = [r for r in results if r["expected_escalation"]]
    correct_handle_rate   = sum(r["routing_hit"] for r in handle_cases)   / len(handle_cases)   if handle_cases   else 0.0
    missed_escalation_rate = sum(r["routing_hit"] for r in escalate_cases) / len(escalate_cases) if escalate_cases else 0.0

    faithfulness_norm  = sum(r["faithfulness"] for r in results) / (n * 5)
    correctness_norm   = sum(r["correctness"]  for r in results) / (n * 5)

    # Steps distribution (only meaningful for agent mode; all zeros in naive mode)
    steps = [r["steps_taken"] for r in results]
    steps_dist = {
        "1":   sum(1 for s in steps if s == 1),
        "2":   sum(1 for s in steps if s == 2),
        "3+":  sum(1 for s in steps if s >= 3),
        "avg": round(sum(steps) / n, 2) if steps else 0.0,
    }

    return {
        "n":                       n,
        "classification_accuracy": round(classification_acc,       4),
        "routing_accuracy":        round(routing_acc,              4),
        "correct_handle_rate":     round(correct_handle_rate,      4),
        "missed_escalation_rate":  round(missed_escalation_rate,   4),
        "n_handle_cases":          len(handle_cases),
        "n_escalation_cases":      len(escalate_cases),
        "faithfulness":            round(faithfulness_norm,        4),
        "correctness":             round(correctness_norm,         4),
        "faithfulness_raw":        round(sum(r["faithfulness"] for r in results) / n, 2),
        "correctness_raw":         round(sum(r["correctness"]  for r in results) / n, 2),
        "steps_distribution":      steps_dist,
    }


# ---------------------------------------------------------------------------
# Stratified eval by intent  (B2.2 stretch)
# ---------------------------------------------------------------------------

def run_stratified_eval(results: list[dict]) -> dict:
    by_intent: dict[str, list] = {}
    for r in results:
        by_intent.setdefault(r["expected_intent"], []).append(r)

    intent_breakdown = {}
    for intent, rows in by_intent.items():
        total          = len(rows)
        correct_class  = sum(r["classification_hit"] for r in rows)
        intent_breakdown[intent] = {
            "total":                    total,
            "classification_correct":   correct_class,
            "classification_accuracy":  round(correct_class / total, 4) if total else 0.0,
            "faithfulness_raw":         round(sum(r["faithfulness"] for r in rows) / total, 2),
            "correctness_raw":          round(sum(r["correctness"]  for r in rows) / total, 2),
        }

    return {"by_intent": intent_breakdown}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_scorecard(scorecard: dict, title: str = "Overall Scorecard") -> None:
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Queries evaluated      : {scorecard.get('n', 0)}")
    print(f"  Classification Acc     : {scorecard.get('classification_accuracy', 0):.1%}")
    print(f"  Routing Accuracy       : {scorecard.get('routing_accuracy', 0):.1%}")
    print(f"    ├─ Correct-Handle    : {scorecard.get('correct_handle_rate', 0):.1%}  "
          f"(n={scorecard.get('n_handle_cases', 0)})")
    print(f"    └─ Missed-Escalation : {scorecard.get('missed_escalation_rate', 0):.1%}  "
          f"(n={scorecard.get('n_escalation_cases', 0)})")
    print(f"  Faithfulness           : {scorecard.get('faithfulness_raw', 0):.2f}/5.0  "
          f"({scorecard.get('faithfulness', 0):.1%})")
    print(f"  Correctness            : {scorecard.get('correctness_raw', 0):.2f}/5.0  "
          f"({scorecard.get('correctness', 0):.1%})")
    dist = scorecard.get("steps_distribution", {})
    if dist.get("avg", 0) > 0:
        print(f"  Steps/query (avg)      : {dist['avg']}  "
              f"[1-step={dist.get('1',0)}  2-step={dist.get('2',0)}  3+={dist.get('3+',0)}]")
    print("=" * width + "\n")


def print_comparison(agent_sc: dict, naive_sc: dict) -> None:
    """Print a side-by-side delta table: agent vs naive pipeline."""
    width = 68
    print("\n" + "=" * width)
    print("  Agent vs Naive Pipeline — Comparison")
    print("=" * width)
    metrics = [
        ("Classification Acc",     "classification_accuracy", ".1%"),
        ("Routing Accuracy",        "routing_accuracy",        ".1%"),
        ("Missed-Escalation Rate",  "missed_escalation_rate",  ".1%"),
        ("Faithfulness (raw/5)",    "faithfulness_raw",        ".2f"),
        ("Correctness (raw/5)",     "correctness_raw",         ".2f"),
    ]
    print(f"  {'Metric':<28}  {'Naive':>8}  {'Agent':>8}  {'Delta':>8}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, key, fmt in metrics:
        naive_val = naive_sc.get(key, 0)
        agent_val = agent_sc.get(key, 0)
        delta     = agent_val - naive_val
        sign      = "+" if delta >= 0 else ""
        n_str  = format(naive_val, fmt)
        a_str  = format(agent_val, fmt)
        d_str  = sign + format(delta, fmt)
        print(f"  {label:<28}  {n_str:>8}  {a_str:>8}  {d_str:>8}")
    dist = agent_sc.get("steps_distribution", {})
    if dist.get("avg", 0) > 0:
        print(f"\n  Agent steps/query: avg={dist['avg']}  "
              f"[1-step={dist.get('1',0)}  2-step={dist.get('2',0)}  3+={dist.get('3+',0)}]")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Novus Support Agent eval harness")
    parser.add_argument("--naive",        action="store_true",
                        help="Use naive pipeline (handle_query) instead of LangGraph agent")
    parser.add_argument("--compare",      action="store_true",
                        help="Run both agent and naive pipeline and print delta table")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save aggregate scores as baseline_scores.json")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Evaluate only first N entries (smoke test)")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress per-entry progress output")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    if args.compare:
        # Run agent first, then naive, print both scorecards + delta
        print(f"\nRunning agent eval on {len(dataset)} queries…\n")
        t0 = time.time()
        agent_results = run_eval(dataset, verbose=not args.quiet, use_agent=True)
        agent_sc = compute_scorecard(agent_results)
        agent_strat = run_stratified_eval(agent_results)
        print(f"Agent eval done in {round(time.time()-t0, 1)}s")

        print(f"\nRunning naive pipeline eval on {len(dataset)} queries…\n")
        t0 = time.time()
        naive_results = run_eval(dataset, verbose=not args.quiet, use_agent=False)
        naive_sc = compute_scorecard(naive_results)
        print(f"Naive eval done in {round(time.time()-t0, 1)}s")

        print_scorecard(agent_sc, title="Novus Support Agent — LangGraph Agent")
        print_scorecard(naive_sc, title="Novus Support Agent — Naive Pipeline (Week 2)")
        print_comparison(agent_sc, naive_sc)

        output = {
            "mode": "compare",
            "agent":   {"scorecard": agent_sc, "stratified": agent_strat, "results": agent_results},
            "naive":   {"scorecard": naive_sc, "results": naive_results},
        }
        RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved → {RESULTS_PATH}")
        return

    use_agent = not args.naive
    mode_label = "LangGraph Agent" if use_agent else "Naive Pipeline (Week 2)"
    print(f"\nRunning eval [{mode_label}] on {len(dataset)} queries…\n")

    t0 = time.time()
    results = run_eval(dataset, verbose=not args.quiet, use_agent=use_agent)
    elapsed = round(time.time() - t0, 1)

    scorecard  = compute_scorecard(results)
    stratified = run_stratified_eval(results)

    print_scorecard(scorecard, title=f"Novus Support Agent — {mode_label}")

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

    output = {"mode": mode_label, "scorecard": scorecard, "stratified": stratified, "results": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {RESULTS_PATH}")

    if args.save_baseline:
        baseline = {"mode": mode_label, "scorecard": scorecard, "stratified": stratified}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Baseline saved → {BASELINE_PATH}")


if __name__ == "__main__":
    main()
