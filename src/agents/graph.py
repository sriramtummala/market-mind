"""
LangGraph orchestration wiring together:
  research_query  -> (routes here if it's a research question)
  compliance_check -> risk_score -> human_approval (interrupt) -> done

This is the "supervisor" pattern: a router node inspects the incoming
request type and directs it to the right sub-agent(s). Trade-related
requests always flow through BOTH compliance and risk agents before
reaching a human-in-the-loop approval gate — no automatic execution.
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.rag import RetrievalIndex, answer_question
from src.agents.compliance_agent import review_trade
from src.agents.risk_agent import score_risk

_index = RetrievalIndex()

class AgentState(TypedDict, total=False):
    request_type: Literal["research", "trade"]
    query: Optional[str]
    trade: Optional[dict]
    research_result: Optional[dict]
    compliance_result: Optional[dict]
    risk_result: Optional[dict]
    human_decision: Optional[str]   # "approve" | "reject" | None (awaiting)
    final_status: Optional[str]

def route(state: AgentState) -> str:
    return "research_node" if state["request_type"] == "research" else "compliance_node"

def research_node(state: AgentState) -> AgentState:
    result = answer_question(state["query"], _index)
    state["research_result"] = result
    state["final_status"] = "RESEARCH_COMPLETE"
    return state

def compliance_node(state: AgentState) -> AgentState:
    state["compliance_result"] = review_trade(state["trade"], _index)
    return state

def risk_node(state: AgentState) -> AgentState:
    state["risk_result"] = score_risk(state["trade"], _index)
    return state

def human_approval_node(state: AgentState) -> AgentState:
    """
    Interrupt point: in the real app (see api.py), execution pauses here and
    returns the compliance + risk results to a human reviewer. The graph is
    resumed with state["human_decision"] set once they respond.
    """
    if state.get("human_decision") == "approve":
        state["final_status"] = "APPROVED_BY_HUMAN"
    elif state.get("human_decision") == "reject":
        state["final_status"] = "REJECTED_BY_HUMAN"
    else:
        state["final_status"] = "AWAITING_HUMAN_APPROVAL"
    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("research_node", research_node)
    graph.add_node("compliance_node", compliance_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("human_approval_node", human_approval_node)

    graph.set_conditional_entry_point(
        route, {"research_node": "research_node", "compliance_node": "compliance_node"}
    )
    graph.add_edge("research_node", END)
    graph.add_edge("compliance_node", "risk_node")
    graph.add_edge("risk_node", "human_approval_node")
    graph.add_edge("human_approval_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_approval_node"])

APP_GRAPH = build_graph()

def run_research(query: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    return APP_GRAPH.invoke({"request_type": "research", "query": query}, config=config)


def submit_trade(trade: dict, thread_id: str):
    """Step 1: run compliance + risk, then PAUSE for human approval."""
    config = {"configurable": {"thread_id": thread_id}}
    return APP_GRAPH.invoke({"request_type": "trade", "trade": trade}, config=config)


def resolve_trade(thread_id: str, decision: str):
    """Step 2: human calls this with 'approve' or 'reject' to resume the graph."""
    config = {"configurable": {"thread_id": thread_id}}
    APP_GRAPH.update_state(config, {"human_decision": decision})
    return APP_GRAPH.invoke(None, config=config)

if __name__ == "__main__":
    print("--- Research path ---")
    print(run_research("What is ACME Robotics' market concentration risk?")["research_result"]["answer"][:200])

    print("\n--- Trade path (pauses for human approval) ---")
    trade = {"ticker": "BEACON", "action": "BUY", "quantity": 500,
              "client_portfolio_value": 40000, "position_value": 12000}
    state = submit_trade(trade, thread_id="demo-1")
    print("Compliance verdict:", state["compliance_result"]["verdict"])
    print("Risk band:", state["risk_result"]["risk_band"])
    print("Status:", state.get("final_status", "AWAITING_HUMAN_APPROVAL (graph paused)"))

    print("\n--- Human approves ---")
    final = resolve_trade("demo-1", "approve")
    print("Final status:", final["final_status"])
