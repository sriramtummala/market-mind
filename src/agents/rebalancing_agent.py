"""
Portfolio rebalancing agent demonstrating three agentic patterns:
  1. ReAct — an explicit Thought -> Action (tool call) -> Observation loop
     instead of a single-shot prompt.
  2. Memory — conversation state persists across turns within a session,
     so follow-up questions don't need to repeat context.
  3. Reflection — the agent's draft recommendation is critiqued against
     policy before being returned; if issues are found, it revises once.
"""

import json
import sys
from src.llm import call_llm_with_tools, call_llm

MAX_REACT_ITERATIONS = 5

TOOLS = [
    {
        "name": "get_portfolio_holdings",
        "description": "Get a client's current portfolio holdings by client ID.",
        "input_schema": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    },
    {
        "name": "get_current_price",
        "description": "Get the current market price for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_option_price",
        "description": "Estimate an option's price using a simplified Black-Scholes model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "strike": {"type": "number"},
                "days_to_expiry": {"type": "integer"},
                "option_type": {"type": "string", "enum": ["call", "put"]},
            },
            "required": ["ticker", "strike", "days_to_expiry", "option_type"],
        },
    },
]

_SYSTEM_PROMPT = """You are a portfolio rebalancing agent. Use the available
tools to look up real holdings and prices before making any recommendation.
Never fabricate a number you could have looked up. Once you have enough
information, give a specific, actionable rebalancing recommendation."""

# --- Mock data stores (used instead of a live brokerage API) ---
_MOCK_HOLDINGS = {
    "client-1": {"ACME": 45.0, "BEACON": 10.0, "HARBOR": 20.0, "cash": 25.0},
}
_MOCK_PRICES = {"ACME": 187.40, "BEACON": 22.15, "HARBOR": 68.90}

# --- Session memory (in-process; swap for Redis/DB in production) ---
_SESSIONS: dict[str, list] = {}

# --- Tool implementations ---
def get_portfolio_holdings(client_id: str) -> dict:
    return _MOCK_HOLDINGS.get(client_id, {"error": "client not found"})


def get_current_price(ticker: str) -> dict:
    price = _MOCK_PRICES.get(ticker.upper())
    return {"ticker": ticker.upper(), "price": price} if price else {"error": "ticker not found"}


def calculate_option_price(ticker: str, strike: float, days_to_expiry: int, option_type: str) -> dict:
    """Simplified Black-Scholes-style estimate — not investment-grade pricing,
    just enough to demonstrate a tool the agent can call mid-reasoning."""
    import math
    spot = _MOCK_PRICES.get(ticker.upper(), 100.0)
    t = days_to_expiry / 365
    vol = 0.35  # flat assumed volatility for demo purposes
    intrinsic = max(0, spot - strike) if option_type == "call" else max(0, strike - spot)
    time_value = spot * vol * math.sqrt(t) * 0.4  # crude time-value approximation
    return {"estimated_price": round(intrinsic + time_value, 2), "spot": spot,
            "note": "Simplified demo pricing model, not investment-grade Black-Scholes."}


_TOOL_IMPL = {
    "get_portfolio_holdings": get_portfolio_holdings,
    "get_current_price": get_current_price,
    "calculate_option_price": calculate_option_price,
}

def _execute_tool(name: str, tool_input: dict) -> dict:
    fn = _TOOL_IMPL.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    return fn(**tool_input)

def react_loop(user_request: str, session_id: str) -> dict:
    """Runs the Thought/Action/Observation loop until the model returns a
    final answer (no more tool calls) or MAX_REACT_ITERATIONS is hit."""
    history = _SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": user_request})

    trace = []
    for _ in range(MAX_REACT_ITERATIONS):
        response = call_llm_with_tools(_SYSTEM_PROMPT, history, TOOLS)

        if response["stop_reason"] != "tool_use" or not response["tool_calls"]:
            history.append({"role": "assistant", "content": response["text"]})
            return {"draft_answer": response["text"], "trace": trace, "session_id": session_id}

        # Replay the assistant's actual tool_use blocks (with their ids) so the
        # follow-up tool_result blocks below can reference them correctly —
        # Anthropic's API only allows "user"/"assistant" roles, and tool
        # results are submitted as a "user" message with tool_result blocks.
        assistant_content = []
        if response["text"]:
            assistant_content.append({"type": "text", "text": response["text"]})
        for call in response["tool_calls"]:
            assistant_content.append({
                "type": "tool_use", "id": call["id"], "name": call["name"], "input": call["input"],
            })
        history.append({"role": "assistant", "content": assistant_content})

        tool_result_content = []
        for call in response["tool_calls"]:
            observation = _execute_tool(call["name"], call["input"])
            trace.append({"action": call["name"], "input": call["input"], "observation": observation})
            tool_result_content.append({
                "type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(observation),
            })
        history.append({"role": "user", "content": tool_result_content})

    return {"draft_answer": "Max iterations reached without a final answer.",
            "trace": trace, "session_id": session_id}

_REFLECTION_SYSTEM_PROMPT = """You are a compliance reviewer critiquing a
draft portfolio rebalancing recommendation. Check it against: (1) no single
position should exceed 25% of portfolio after the change, (2) the
recommendation must be specific and actionable. Output ONLY JSON:
{"issues_found": ["..."], "approved": true|false}"""


def reflect_on_draft(draft: str) -> dict:
    raw = call_llm(_REFLECTION_SYSTEM_PROMPT, draft, role="reflection")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"issues_found": ["Reflection output was not valid JSON"], "approved": False}
    

def get_rebalancing_recommendation(user_request: str, session_id: str = "default") -> dict:
    """Full pipeline: ReAct tool-calling -> draft -> Reflection -> revise if needed."""
    result = react_loop(user_request, session_id)
    draft = result["draft_answer"]

    critique = reflect_on_draft(draft)
    if critique.get("approved"):
        return {"final_answer": draft, "trace": result["trace"], "revised": False, "critique": critique}

    # Revise once, feeding the critique back in
    revision_request = (f"REVISED REQUEST — your previous draft had issues: "
                         f"{critique.get('issues_found')}. Original draft: {draft}. "
                         f"Please provide a corrected recommendation.")
    revised = react_loop(revision_request, session_id)
    return {
        "final_answer": revised["draft_answer"],
        "trace": result["trace"] + revised["trace"],
        "revised": True,
        "critique": critique,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = get_rebalancing_recommendation(
        "Client client-1 is overweight tech. Recommend a rebalancing action.",
        session_id="demo-session",
    )
    print("--- ReAct trace ---")
    for step in result["trace"]:
        print(f"Action: {step['action']}({step['input']}) -> Observation: {step['observation']}")
    print("\n--- Reflection ---")
    print("Revised after critique:", result["revised"])
    print("Critique:", result["critique"])
    print("\n--- Final recommendation ---")
    print(result["final_answer"])
