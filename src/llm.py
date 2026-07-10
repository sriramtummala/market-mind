"""
llm.py — thin LLM wrapper.

Uses the real Anthropic API if ANTHROPIC_API_KEY is set in the environment.
Falls back to a deterministic MOCK mode otherwise, so the whole pipeline
(retrieval -> agents -> orchestration -> guardrails -> eval) can be built,
run, and demoed end-to-end without needing a paid API key on day 1.

Swap MOCK_MODE off once you plug in your own key — nothing else in the
codebase needs to change.
"""
import os
import json

from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.environ.get("ANTHROPIC_API_KEY") is None

if not MOCK_MODE:
    import anthropic
    _client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"


def call_llm(system: str, user: str, max_tokens: int = 800, role: str = "research") -> str:
    """Single entry point every agent uses to talk to the LLM.

    `role` is only used to pick the right canned response in MOCK_MODE
    (one of: "research", "compliance", "risk", "extraction"). Real API
    calls ignore it.
    """
    if MOCK_MODE:
        return _mock_response(role, user)

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_llm_with_tools(system: str, messages: list, tools: list, role: str = "research") -> dict:
    """
    Tool-calling entry point for ReAct-style agents (Day 9).
    Returns {"stop_reason": "tool_use"|"end_turn", "tool_calls": [...], "text": "..."}

    In MOCK_MODE, returns a deterministic scripted sequence of tool calls
    (see _mock_tool_sequence) so the ReAct loop is fully testable offline.
    Real mode uses Anthropic's native tool-use API.
    """
    if MOCK_MODE:
        return _mock_tool_response(messages)

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=tools,
    )
    tool_calls = [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in resp.content if b.type == "tool_use"
    ]
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"stop_reason": resp.stop_reason, "tool_calls": tool_calls, "text": text}


def _mock_response(role: str, user: str = "") -> str:
    """
    Deterministic stand-in used when no API key is configured.
    Good enough to exercise every downstream code path (JSON parsing,
    citation checks, guardrails) during local development / CI.
    """
    if role == "compliance":
        return json.dumps({
            "verdict": "FLAGGED_FOR_REVIEW",
            "reasoning": "Mock mode: at least one policy match was found in the "
                         "retrieved context; a real LLM call would give a full "
                         "policy-by-policy analysis here.",
            "policies_triggered": ["Policy 1", "Policy 2"],
        })
    if role == "risk":
        return json.dumps({
            "risk_score": 62,
            "risk_band": "MEDIUM-HIGH",
            "rationale": "Mock mode: score derived from placeholder heuristics "
                         "(concentration + volatility proxy). Replace with real "
                         "LLM + quant scoring once API key is set.",
        })
    if role == "extraction":
        return _mock_extract_fields(user)
    if role == "reflection":
        return _mock_reflection(user)
    # default: research/RAG answer
    return ("Mock mode response (no ANTHROPIC_API_KEY set). Based on the "
            "retrieved context above, here is a placeholder synthesized answer "
            "citing [see sources list]. Set ANTHROPIC_API_KEY to get a real, "
            "grounded answer from Claude.")


def _mock_extract_fields(user: str) -> str:
    """
    Regex-based field extraction used in MOCK_MODE so the document
    intelligence demo (Day 8) genuinely produces correct structured data
    from the sample documents without needing an API key. A real LLM call
    handles messier/less structured real-world documents far better than
    this — this is a mock stand-in, not a production extraction strategy.
    """
    import re

    def find(pattern, text, cast=str):
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).strip()
        if cast is float:
            return float(val.replace(",", "").replace("USD", "").strip())
        return val

    text = user

    if "trade_date" in text and "counterparty" in text:
        result = {
            "trade_date": find(r"Trade Date:\s*([\d-]+)", text),
            "counterparty": find(r"Counterparty:\s*(.+)", text),
            "instrument": find(r"Instrument:\s*(.+)", text),
            "notional_amount": find(r"Notional Amount:\s*(?:USD\s*)?([\d,]+)", text, float),
            "settlement_date": find(r"Settlement Date:\s*([\d-]+)", text),
        }
        return json.dumps(result)

    if "party_a" in text and "party_b" in text:
        result = {
            "party_a": find(r"Party A:\s*(.+)", text),
            "party_b": find(r"Party B:\s*(.+)", text),
            "agreement_date": find(r"Agreement Date:\s*([\d-]+)", text),
            "governing_law": find(r"Governing Law:\s*(.+)", text),
            "threshold_amount": find(r"Threshold Amount:\s*(?:USD\s*)?([\d,]+)", text, float),
        }
        return json.dumps(result)

    if "underlying" in text and "rows" in text:
        # Tesseract frequently OCRs simple tables column-by-column rather
        # than row-by-row (no visible gridlines to anchor row segmentation).
        # Real production systems handle this with layout-aware table
        # detection (e.g. pdfplumber table extraction, or a vision-capable
        # LLM call on the page image) rather than plain regex — this
        # column-reassembly approach is a pragmatic mock-mode stand-in that
        # demonstrates the exact failure mode you'll want to design around.
        underlying = find(r"\(([A-Z]{2,6})\)", text) or find(r"([A-Z]{2,6})\s+OPTIONS CHAIN", text)
        expiration = find(r"EXP\s+([\d-]+)", text)

        strikes = [float(x) for x in re.findall(r"STRIKE\s*\n((?:[\d.]+\n?)+)", text, re.MULTILINE)
                   for x in re.findall(r"[\d.]+", x)] if re.search(r"STRIKE", text) else []
        types = re.findall(r"\b(CALL|PUT)\b", text)
        bids, asks, volumes = [], [], []
        bid_block = re.search(r"BID\s*\n\n((?:[\d.]+\n?)+)", text)
        ask_block = re.search(r"ASK\s*\n\n((?:[\d.]+\n?)+)", text)
        vol_block = re.search(r"VOLUME\s*\n((?:\d+\n?)+)", text)
        if bid_block:
            bids = [float(x) for x in re.findall(r"[\d.]+", bid_block.group(1))]
        if ask_block:
            asks = [float(x) for x in re.findall(r"[\d.]+", ask_block.group(1))]
        if vol_block:
            volumes = [int(x) for x in re.findall(r"\d+", vol_block.group(1))]

        rows = []
        n = min(len(strikes), len(types), len(bids), len(asks), len(volumes)) if all(
            [strikes, types, bids, asks, volumes]) else 0
        for i in range(n):
            rows.append({"strike": strikes[i], "type": types[i], "bid": bids[i],
                         "ask": asks[i], "volume": volumes[i]})
        return json.dumps({"underlying": underlying, "expiration": expiration, "rows": rows})

    return json.dumps({"error": "mock extractor did not recognize schema"})


def _mock_reflection(user: str) -> str:
    """Mock critique for the Reflection pattern (Day 9). Deliberately flags
    an issue the first time so the ReAct/Reflection loop visibly revises
    its draft — demonstrating the pattern rather than always passing."""
    if "REVISED" in user.upper():
        return json.dumps({"issues_found": [], "approved": True})
    return json.dumps({
        "issues_found": ["Position concentration remains above the 25% policy "
                         "limit after the proposed trade — reduce further."],
        "approved": False,
    })


def _mock_tool_response(messages: list) -> dict:
    """
    Deterministic scripted ReAct sequence for MOCK_MODE (Day 9): the first
    call requests portfolio holdings, the second requests a price quote,
    the third returns a final answer. Real mode lets the model decide tool
    calls dynamically; this just proves the loop mechanics work end to end.
    """
    num_tool_results = sum(1 for m in messages if m.get("role") == "tool")
    if num_tool_results == 0:
        return {"stop_reason": "tool_use", "text": "",
                "tool_calls": [{"id": "call_1", "name": "get_portfolio_holdings",
                                 "input": {"client_id": "client-1"}}]}
    if num_tool_results == 1:
        return {"stop_reason": "tool_use", "text": "",
                "tool_calls": [{"id": "call_2", "name": "get_current_price",
                                 "input": {"ticker": "ACME"}}]}
    return {"stop_reason": "end_turn", "tool_calls": [],
            "text": "Based on current holdings and price, recommend reducing "
                    "ACME position by 10% to bring concentration within policy limits."}
