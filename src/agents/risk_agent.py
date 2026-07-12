"""
Combines a simple quantative concentration % of portfolio with an LLM
quantative assement grounded in the issuer's filling risk factors, to produce a blended risk score.
This mirros "quant score + LLM judgement" patterns used in real portfolio risk tooling
"""

import json
from src.llm import call_llm
from src.rag import RetrievalIndex

_SYSTEM_PROMPT = """You are a portfolio risk assesment agent. you are given a proposed position,
a computed conecentration percentage, and experts from the issuer's SEC filing risk factors.
Blend the quantative concentration signal with the qualitative disclosure risks to procude a 0 - 100 risk score
 Output ONLY JSON:
{"risk_score": <int 0-100>, "risk_band": "LOW"|"MEDIUM"|"MEDIUM-HIGH"|"HIGH",
 "rationale": "concise reasoning referencing filenames and the concentration %"}
"""
def compute_concentration(trade: dict) -> dict:
    portfolio_value = trade.get("client_portfolio_value", 0)
    position_value = trade.get("position_value", 0)
    if portfolio_value <= 0:
        return 0.0
    return round(100 * position_value / portfolio_value, 1)

def score_risk(trade: dict, index: RetrievalIndex) -> dict:
    concentration_pct = compute_concentration(trade)
    filing_hits = index.search(f"{trade.get('ticker')} risk factors", k=3)
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in filing_hits)
    user_prompt = (
            f"TRADE: {json.dumps(trade)}\n"
            f"COMPUTED CONCENTRATION: {concentration_pct}% of client portfolio\n\n"
            f"ISSUER RISK FACTOR EXCERPTS:\n{context}\n\n"
            "Produce the blended risk score."
        )

    raw = call_llm(_SYSTEM_PROMPT, user_prompt, role="risk")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"risk_score": 75, "risk_band": "HIGH",
                   "rationale": f"Non-JSON LLM output, defaulting to conservative "
                                f"HIGH risk band. Raw: {raw[:200]}"}
    result["computed_concentration_pct"] = concentration_pct
    result["sources_considered"] = [h["source"] for h in filing_hits]
    return result


if __name__ == "__main__":
    idx = RetrievalIndex()
    trade = {"ticker": "BEACON", "action": "BUY", "quantity": 500,
              "client_portfolio_value": 40000, "position_value": 12000}
    print(json.dumps(score_risk(trade, idx), indent=2))
