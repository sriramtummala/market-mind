"""
FastAPI microservice exposing:
  POST /research        -> RAG research question, guarded + cited
  POST /trade/submit     -> runs compliance + risk agents, pauses for approval
  POST /trade/resolve    -> human approves/rejects, resumes the graph

Run: uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.guardrails import guard_user_input, check_citation_grounding
from src.agents.graph import run_research, submit_trade, resolve_trade

app = FastAPI(title="MarketMind API", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str
    thread_id: str = "default"

class TradeRequest(BaseModel):
    ticker: str
    action: str
    quantity: int
    client_portfolio_value: float
    position_value: float
    thread_id: str


class ResolveRequest(BaseModel):
    thread_id: str
    decision: str  # "approve" | "reject"

@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/research")
def research(req: ResearchRequest):
    guard = guard_user_input(req.query)
    if guard["blocked"]:
        return {"error": "Request blocked by guardrails", "reason": guard["block_reason"]}
    state = run_research(guard["safe_text"], thread_id=req.thread_id)
    result = state["research_result"]
    grounding = check_citation_grounding(
        result["answer"], allowed_sources=[s["source"] for s in result["sources"]]
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "pii_redacted_from_query": guard["pii_found"],
        "grounding_check": grounding,
    }
@app.post("/trade/submit")
def trade_submit(req: TradeRequest):
    trade = req.dict(exclude={"thread_id"})
    state = submit_trade(trade, thread_id=req.thread_id)
    return {
        "compliance": state["compliance_result"],
        "risk": state["risk_result"],
        "status": "AWAITING_HUMAN_APPROVAL",
        "thread_id": req.thread_id,
    }


@app.post("/trade/resolve")
def trade_resolve(req: ResolveRequest):
    final = resolve_trade(req.thread_id, req.decision)
    return {"final_status": final["final_status"]}