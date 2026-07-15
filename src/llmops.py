"""
llmops.py — Day 11: MLflow experiment tracking for eval runs and a
compliance-agent system-prompt A/B test.

Uses a local, file-based MLflow tracking store (mlruns/, already
git-ignored) so this runs fully offline — no MLflow server, no extra
credentials. This is deliberately just the MLflow slice of the full
enterprise LLMOps stack; LangSmith tracing, TruLens feedback functions, and
real RAGAS LLM-judge metrics need external API keys/accounts and are a
follow-up, not implemented here.

Run: python -m src.llmops
"""
import json
import os

import mlflow

from src.eval import run_eval
from src.rag import RetrievalIndex
from src.llm import call_llm
from src.agents.compliance_agent import _SYSTEM_PROMPT as PROMPT_VARIANT_A

MLFLOW_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("marketmind")


def log_eval_run(run_name: str = "rag_eval") -> dict:
    """Runs the Day 6 offline RAG eval harness and logs its summary metrics
    as an MLflow run, so retrieval/grounding quality can be compared across
    commits or prompt changes over time instead of just read off stdout once."""
    summary = run_eval()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("num_questions", summary["num_questions"])
        mlflow.log_metric("avg_context_precision", summary["avg_context_precision"])
        mlflow.log_metric("avg_keyword_recall", summary["avg_keyword_recall"])
        mlflow.log_metric("passed", int(summary["passed"]))
        mlflow.log_dict(summary, "eval_summary.json")
    return summary


# --- Compliance agent system-prompt A/B test ---

PROMPT_VARIANT_B = """You are a senior broker-dealer trading compliance officer
with 20 years of experience. Carefully read the proposed trade and the
attached policy/filing excerpts. Reason step by step about every policy that
could apply, being conservative: if a policy's applicability is ambiguous,
treat it as triggered rather than waived. Output ONLY a JSON object of this
exact shape:
{
  "verdict": "APPROVED" | "FLAGGED_FOR_REVIEW" | "BLOCKED",
  "policies_triggered": ["Policy N: short description", ...],
  "reasoning": "concise step-by-step reasoning citing filenames"
}
Never approve a trade automatically if Policy 8 (human-in-the-loop) or any
BLOCKED-level policy applies. Do not include any text outside the JSON."""

# Ground-truth verdicts derived from data/policies/trading_compliance_policy.md,
# not invented: Beacon's $612M accumulated deficit exceeds Policy 2's $500M
# going-concern threshold; Harbor is a well-capitalized bank (CET1 10.8%,
# above minimum) at only 5% portfolio concentration, well under Policy 3's
# 25% limit, with no going-concern disclosure.
_AB_TEST_CASES = [
    {
        "trade": {"ticker": "BEACON", "action": "BUY", "quantity": 500,
                  "client_portfolio_value": 40000, "position_value": 12000},
        "expected_verdicts": {"FLAGGED_FOR_REVIEW", "BLOCKED"},
        "why": "Policy 2 going-concern threshold ($500M) triggered by Beacon's $612M accumulated deficit",
    },
    {
        "trade": {"ticker": "HARBOR", "action": "BUY", "quantity": 20,
                  "client_portfolio_value": 40000, "position_value": 2000},
        "expected_verdicts": {"APPROVED"},
        "why": "Well-capitalized bank, 5% concentration — no policy triggers expected",
    },
]


def _review_with_prompt(system_prompt: str, trade: dict, index: RetrievalIndex) -> dict:
    """Same retrieval + prompting shape as
    src/agents/compliance_agent.review_trade, parameterized on system_prompt
    so both variants can be scored without touching the production agent."""
    query = (f"trading policy rules relevant to {trade.get('ticker')} "
             f"market cap going concern concentration limits")
    policy_hits = index.search(query, k=4)
    filing_query = f"{trade.get('ticker')} market capitalization going concern risk"
    filing_hits = index.search(filing_query, k=3)
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in (policy_hits + filing_hits))
    user_prompt = (f"PROPOSED TRADE:\n{json.dumps(trade, indent=2)}\n\n"
                   f"RELEVANT POLICY & FILING EXCERPTS:\n{context}\n\n"
                   "Evaluate this trade against policy.")
    raw = call_llm(system_prompt, user_prompt, role="compliance")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "FLAGGED_FOR_REVIEW", "policies_triggered": [],
                "reasoning": f"non-JSON output: {raw[:200]}"}


def _score_variant(system_prompt: str, index: RetrievalIndex) -> dict:
    results = []
    correct = 0
    for case in _AB_TEST_CASES:
        verdict_result = _review_with_prompt(system_prompt, case["trade"], index)
        verdict = verdict_result.get("verdict")
        is_correct = verdict in case["expected_verdicts"]
        correct += int(is_correct)
        results.append({"ticker": case["trade"]["ticker"], "verdict": verdict,
                         "expected": sorted(case["expected_verdicts"]), "correct": is_correct})
    return {"accuracy": round(correct / len(_AB_TEST_CASES), 3), "cases": results}


def run_ab_test() -> dict:
    """A/B tests two compliance-agent system prompts against trades with
    known-correct verdicts (per actual policy thresholds), logs both
    variants to MLflow as nested runs, and picks a winner by accuracy."""
    index = RetrievalIndex()
    variants = {"variant_a_production": PROMPT_VARIANT_A,
                "variant_b_senior_conservative": PROMPT_VARIANT_B}

    scored = {}
    with mlflow.start_run(run_name="compliance_prompt_ab_test"):
        for name, prompt in variants.items():
            result = _score_variant(prompt, index)
            scored[name] = result
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("prompt_variant", name)
                mlflow.log_metric("accuracy", result["accuracy"])
                mlflow.log_dict(result, f"{name}_cases.json")

        winner = max(scored, key=lambda k: scored[k]["accuracy"])
        mlflow.log_param("winner", winner)
        mlflow.log_metric("winner_accuracy", scored[winner]["accuracy"])

    return {"scored": scored, "winner": winner}


if __name__ == "__main__":
    print("--- Logging RAG eval run to MLflow ---")
    eval_summary = log_eval_run()
    print(f"avg_context_precision={eval_summary['avg_context_precision']} "
          f"avg_keyword_recall={eval_summary['avg_keyword_recall']} "
          f"passed={eval_summary['passed']}")

    print("\n--- Running compliance-agent prompt A/B test ---")
    ab_result = run_ab_test()
    for name, result in ab_result["scored"].items():
        print(f"{name}: accuracy={result['accuracy']}")
    print(f"\nWinner: {ab_result['winner']}")
    print(f"\nMLflow tracking data written to: {MLFLOW_DB_PATH}")
