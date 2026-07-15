"""
DeepEval assertion-style test suite.

These check structural/business invariants of agent outputs — valid verdict
enums, score bounds, no fabricated citations, the Policy 8 human-approval
guardrail — rather than open-ended answer quality. That's deliberate: they
must pass identically whether ANTHROPIC_API_KEY is set (real Claude) or not
(MOCK_MODE, which is what CI runs under, since ci.yml sets no API key). A
metric that judged wording quality would need a real LLM judge and behave
differently across those two modes.
"""
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

from src.rag import RetrievalIndex, answer_question
from src.agents.compliance_agent import review_trade
from src.agents.risk_agent import score_risk
from src.guardrails import check_citation_grounding


class PredicateMetric(BaseMetric):
    """A deterministic, non-LLM DeepEval metric: wraps a plain Python
    predicate instead of calling an LLM judge, so these tests are free
    (no API calls) and reproducible in both MOCK_MODE and real mode."""

    def __init__(self, name: str, predicate):
        self.name = name
        self.predicate = predicate
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        passed, reason = self.predicate(test_case)
        self.score = 1.0 if passed else 0.0
        self.reason = reason
        self.success = passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


_index = RetrievalIndex()
_TRADE = {"ticker": "BEACON", "action": "BUY", "quantity": 500,
          "client_portfolio_value": 40000, "position_value": 12000}


def _check(name: str, tc: LLMTestCase, predicate):
    assert_test(tc, [PredicateMetric(name, predicate)])


def test_research_answer_has_no_fabricated_citations():
    result = answer_question("What are ACME Robotics' main risk factors?", _index)
    tc = LLMTestCase(
        input="What are ACME Robotics' main risk factors?",
        actual_output=result["answer"],
        context=[s["source"] for s in result["sources"]],
    )
    _check("no_fabricated_citations", tc, lambda tc: (
        not check_citation_grounding(tc.actual_output, allowed_sources=tc.context)["has_fabrication"],
        "answer cited a source that was never retrieved",
    ))


def test_research_answer_within_length_budget():
    result = answer_question("What is Beacon Biotech's cash runway?", _index)
    tc = LLMTestCase(input="What is Beacon Biotech's cash runway?", actual_output=result["answer"])
    _check("max_length_600_words", tc, lambda tc: (
        len(tc.actual_output.split()) <= 600,
        f"answer was {len(tc.actual_output.split())} words, expected <= 600",
    ))


def test_research_grounded_when_sources_found():
    result = answer_question("What is Harbor Financial's CET1 capital ratio?", _index)
    tc = LLMTestCase(input="What is Harbor Financial's CET1 capital ratio?", actual_output=result["answer"])
    _check("grounded_flag_true", tc, lambda tc: (
        result["grounded"] is True, "expected grounded=True when retrieval returned hits",
    ))


def test_compliance_verdict_is_valid_enum():
    result = review_trade(_TRADE, _index)
    tc = LLMTestCase(input=str(_TRADE), actual_output=result["verdict"])
    _check("valid_verdict_enum", tc, lambda tc: (
        tc.actual_output in {"APPROVED", "FLAGGED_FOR_REVIEW", "BLOCKED"},
        f"unexpected verdict: {tc.actual_output!r}",
    ))


def test_compliance_always_requires_human_approval():
    result = review_trade(_TRADE, _index)
    tc = LLMTestCase(input=str(_TRADE), actual_output=str(result["human_approval_required"]))
    _check("policy_8_human_in_the_loop", tc, lambda tc: (
        result["human_approval_required"] is True,
        "Policy 8 hard guardrail was bypassed — human_approval_required was not True",
    ))


def test_risk_score_within_bounds():
    result = score_risk(_TRADE, _index)
    tc = LLMTestCase(input=str(_TRADE), actual_output=str(result["risk_score"]))
    _check("risk_score_0_to_100", tc, lambda tc: (
        0 <= result["risk_score"] <= 100, f"risk_score out of bounds: {result['risk_score']}",
    ))


def test_risk_band_is_valid_enum():
    result = score_risk(_TRADE, _index)
    tc = LLMTestCase(input=str(_TRADE), actual_output=result["risk_band"])
    _check("valid_risk_band_enum", tc, lambda tc: (
        tc.actual_output in {"LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"},
        f"unexpected risk_band: {tc.actual_output!r}",
    ))


def test_beacon_concentration_correctly_computed():
    """$12,000 position / $40,000 portfolio = 30% — pure arithmetic, must
    hold regardless of what the LLM says."""
    result = score_risk(_TRADE, _index)
    tc = LLMTestCase(input=str(_TRADE), actual_output=str(result["computed_concentration_pct"]))
    _check("concentration_math", tc, lambda tc: (
        result["computed_concentration_pct"] == 30.0,
        f"expected 30.0, got {result['computed_concentration_pct']}",
    ))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
