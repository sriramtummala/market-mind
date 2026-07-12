"""
Responsible AI guardrails:
  - PII redaction (account numbers, SSNs, emails, phone numbers) before
    anything is sent to the LLM or logged.
  - Prompt injection detection on user input (heuristic pattern match +
    an explicit test suite in tests/test_guardrails.py).
  - Citation/grounding check on LLM output (used as a lightweight
    hallucination guard alongside the RAGAS eval in Day 6).
"""
import re

_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "account_number": re.compile(r"\b(?:acct|account)[\s#:]*\d{6,12}\b", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

_INJECTION_MARKERS = [
    "ignore previous instructions", "ignore all previous", "disregard the system prompt",
    "you are now", "act as if you have no restrictions", "reveal your system prompt",
    "print your instructions", "jailbreak", "developer mode",
]

def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    redacted = text
    for label, pattern in _PATTERNS.items():
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted, found


def detect_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)

def check_citation_grounding(answer: str, allowed_sources: list[str]) -> dict:
    """
    Lightweight hallucination guard: every bracketed [filename] citation in
    the answer must correspond to a source we actually retrieved. Flags
    fabricated citations.
    """
    cited = set(re.findall(r"\[([\w\-. ]+?\.(?:txt|md|pdf))\]", answer))
    fabricated = cited - set(allowed_sources)
    return {
        "cited_sources": list(cited),
        "fabricated_citations": list(fabricated),
        "has_fabrication": len(fabricated) > 0,
    }


def guard_user_input(text: str) -> dict:
    """Run all input-side guardrails before a request reaches any agent."""
    redacted, pii_found = redact_pii(text)
    injection_flagged = detect_prompt_injection(text)
    return {
        "safe_text": redacted,
        "pii_found": pii_found,
        "blocked": injection_flagged,
        "block_reason": "prompt_injection_detected" if injection_flagged else None,
    }
