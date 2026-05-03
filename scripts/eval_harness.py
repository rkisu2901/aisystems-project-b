"""
Project B Evaluation Harness — Sessions 1 & 2 Starter

4-dimensional eval for the support pipeline:

SESSION 1 functions (implement during Session 1 homework):
  1. check_classification() — did it identify the right intent?
  2. check_routing() — should this have been escalated?
  3. judge_faithfulness() — is the answer grounded in context?
  4. judge_correctness() — does it match the expected answer?
  5. run_eval() — orchestrate and produce scorecard

SESSION 2 functions (implement during Session 2 homework):
  6. run_stratified_eval() — break down by intent and difficulty
  7. attach_langfuse_scores() — attach all 4 dimensions to LangFuse traces
  8. save_baseline() — lock current scores as regression anchor

Run: python scripts/eval_harness.py
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)

# Import pipeline once implemented
# from support_pipeline import handle_query


# =========================================================================
# GOLDEN DATASET
# =========================================================================

def load_golden_dataset():
    """Load Project B's golden dataset."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found for Project B.")
        return []
    with open(path) as f:
        return json.load(f)


# =========================================================================
# SESSION 1: CLASSIFICATION METRICS
# =========================================================================

def check_classification(predicted_intent, expected_intent):
    """
    Did the system classify the query correctly?
    Returns True/False.

    Hint: direct string equality check.

    TODO: Implement in Session 1 homework.
    """
    pass


# =========================================================================
# SESSION 1: ROUTING METRICS
# =========================================================================

def check_routing(predicted_escalation, expected_escalation):
    """
    Should this query have been escalated to a human?
    Did the system make the right routing decision?
    Returns True/False.

    Note: the naive pipeline NEVER escalates (always False).
    So this will be True only for queries where expected_escalation=False.
    That 0% score on escalation cases IS the correct baseline — it's what Week 4 fixes.

    TODO: Implement in Session 1 homework.
    """
    pass


# =========================================================================
# SESSION 1: GENERATION METRICS
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    LLM-as-judge: Is the answer grounded in context?
    Returns: {"score": 1-5, "reason": "explanation"}

    Same pattern as Project A — identical judge prompt.

    TODO: Implement in Session 1 homework.
    """
    pass


def judge_correctness(query, answer, expected_answer):
    """
    LLM-as-judge: Does the answer match the expected answer?
    Returns: {"score": 1-5, "reason": "explanation"}

    TODO: Implement in Session 1 homework.
    """
    pass


# =========================================================================
# SESSION 1: EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run 4-dimensional eval:
    1. Classification accuracy — per intent category
    2. Retrieval quality — hit rate
    3. Response quality — faithfulness + correctness
    4. Routing accuracy — predicted vs expected escalation

    Important: compute routing separately for:
      - Queries where expected_escalation=True  (system should escalate)
      - Queries where expected_escalation=False (system should handle)

    TODO: Implement in Session 1 homework.
    """
    pass


# =========================================================================
# SESSION 2: STRATIFIED EVALUATION
# =========================================================================

def run_stratified_eval(results):
    """
    Break down scores by expected_intent (classification accuracy per intent)
    and by difficulty (correctness per difficulty level).

    Key insight: classification might be 90% overall but 0% on "membership" queries.
    Stratification surfaces this.

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# SESSION 2: LANGFUSE SCORE ATTACHMENT
# =========================================================================

def attach_langfuse_scores(trace_id, classification_correct, retrieval_hit,
                            faithfulness_result, correctness_result, routing_correct):
    """
    Attach all 4 eval dimensions to a LangFuse trace.

    Scores to attach:
      - "classification_correct": 1.0 or 0.0
      - "retrieval_hit": 1.0 or 0.0
      - "faithfulness": faithfulness_result["score"] / 5
      - "correctness": correctness_result["score"] / 5
      - "routing_correct": 1.0 or 0.0

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# SESSION 2: SAVE BASELINE
# =========================================================================

def save_baseline(summary_scores):
    """
    Save current Project B scores as baseline_scores.json.
    Include all 4 dimensions in the baseline.

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--category", type=str)
    args = parser.parse_args()

    print("Project B eval harness skeleton loaded.")
    print()
    print("Session 1 functions: check_classification, check_routing,")
    print("                     judge_faithfulness, judge_correctness, run_eval")
    print()
    print("Session 2 functions: run_stratified_eval, attach_langfuse_scores, save_baseline")
    print()
    print("Note: check_routing baseline will show 0% on escalation cases.")
    print("That is correct — it shows what Week 4 needs to fix.")
