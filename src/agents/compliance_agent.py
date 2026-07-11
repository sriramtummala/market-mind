"""
Given a proposed trade, retrieves relevant policy clauses (RAG over the policy corpus)
plus relavent filing facts and asks the llm to reason step by step (COT) about the which policies
are triggered. returns a structured verdict the orchestrator can act on
"""
import json
from src.llm import call_llm
from src.rag import RetrievalIndex

_SYSTEM_PROMPT = """You are a broker-dealer trading compliance agent. You are
given a proposed trade and relevant excerpts from the firm's internal policy
handbook and the issuer's SEC filing. Think step by step (chain-of-thought)
through each potentially relevant policy, then output ONLY a JSON object with
this exact shape:
{
  "verdict": "APPROVED" | "FLAGGED_FOR_REVIEW" | "BLOCKED",
  "policies_triggered": ["Policy N: short description", ...],
  "reasoning": "concise step-by-step reasoning citing filenames"
}
Never approve a trade automatically if Policy 8 (human-in-the-loop) or any
BLOCKED-level policy applies. Do not include any text outside the JSON."""


def review_trade(trade: dict, index: RetrievalIndex) -> dict:
    """
    trade example:
    {
      "ticker": "BEACON", "action": "BUY", "quantity": 500,
      "client_portfolio_value": 40000, "position_value": 12000
    }
    """
    query = (f"trading policy rules relevant to {trade.get('ticker')} "
             f"market cap going concern concentration limits")
    policy_hits = index.search(query, k=4)

    filing_query = f"{trade.get('ticker')} market capitalization going concern risk"
    filing_hits = index.search(filing_query, k=3)

    context = "\n\n".join(
        f"[{h['source']}]\n{h['text']}" for h in (policy_hits + filing_hits)
    )

    user_prompt = (
        f"PROPOSED TRADE:\n{json.dumps(trade, indent=2)}\n\n"
        f"RELEVANT POLICY & FILING EXCERPTS:\n{context}\n\n"
        "Evaluate this trade against policy."
    )

    raw = call_llm(_SYSTEM_PROMPT, user_prompt, role="compliance")

    raw = call_llm(_SYSTEM_PROMPT, user_prompt, role="compliance")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "verdict": "FLAGGED_FOR_REVIEW",
            "policies_triggered": [],
            "reasoning": f"LLM output was not valid JSON, defaulting to human "
                         f"review for safety. Raw output: {raw[:200]}",
        }

    # Hard guardrail regardless of what the LLM says: Policy 8 always applies.
    result["human_approval_required"] = True
    result["sources_considered"] = [h["source"] for h in (policy_hits + filing_hits)]
    return result


if __name__ == "__main__":
    idx = RetrievalIndex()
    trade = {
        "ticker": "BEACON", "action": "BUY", "quantity": 500,
        "client_portfolio_value": 40000, "position_value": 12000,
    }
    print(json.dumps(review_trade(trade, idx), indent=2))
