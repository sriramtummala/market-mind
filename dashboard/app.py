"""
dashboard/app.py — Day 7

Simple "trading desk assistant" dashboard simulating what a compliance
officer / trader would see. Talks directly to the API layer (src/api.py)
so it exercises the exact same guarded, agent-orchestrated code path a
production frontend would.

Run: streamlit run dashboard/app.py
(make sure `uvicorn src.api:app --port 8000` is running in another terminal)
"""
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="MarketMind — Trading Desk Assistant", layout="wide")
st.title("MarketMind — Agentic Trading Desk Assistant")
st.caption("Demo build — synthetic filings & policies only. Not investment advice.")

tab_research, tab_trade = st.tabs(["📊 Market Research", "✅ Trade Compliance Review"])

with tab_research:
    st.subheader("Ask a research question")
    query = st.text_input("e.g. What are ACME Robotics' main risk factors?")
    if st.button("Ask", key="research_btn") and query:
        with st.spinner("Retrieving and generating grounded answer..."):
            resp = requests.post(f"{API_BASE}/research", json={"query": query})
            data = resp.json()
        if data.get("error"):
            st.error(f"{data['error']}: {data['reason']}")
        else:
            st.markdown("**Answer:**")
            st.write(data["answer"])
            st.markdown("**Sources retrieved:**")
            st.table(data["sources"])
            if data["grounding_check"]["has_fabrication"]:
                st.warning(f"⚠️ Potential fabricated citation(s): "
                           f"{data['grounding_check']['fabricated_citations']}")
            else:
                st.success("✅ All citations grounded in retrieved sources.")

with tab_trade:
    st.subheader("Submit a proposed trade for compliance + risk review")
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Ticker", ["ACME", "BEACON", "HARBOR"])
        action = st.selectbox("Action", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, value=500)
    with col2:
        portfolio_value = st.number_input("Client Portfolio Value ($)", min_value=1.0, value=40000.0)
        position_value = st.number_input("Position Value ($)", min_value=1.0, value=12000.0)

    thread_id = st.text_input("Thread ID (unique per trade)", value=f"{ticker}-{action}-1")

    if st.button("Submit for Review"):
        payload = {"ticker": ticker, "action": action, "quantity": quantity,
                   "client_portfolio_value": portfolio_value, "position_value": position_value,
                   "thread_id": thread_id}
        with st.spinner("Running compliance + risk agents..."):
            resp = requests.post(f"{API_BASE}/trade/submit", json=payload)
            data = resp.json()
        st.session_state["pending_thread"] = thread_id

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Compliance Verdict")
            verdict = data["compliance"]["verdict"]
            color = {"APPROVED": "green", "FLAGGED_FOR_REVIEW": "orange", "BLOCKED": "red"}.get(verdict, "gray")
            st.markdown(f":{color}[**{verdict}**]")
            st.write(data["compliance"]["reasoning"])
            st.write("Policies triggered:", data["compliance"].get("policies_triggered", []))
        with c2:
            st.markdown("### Risk Score")
            st.metric("Risk Band", data["risk"]["risk_band"], f"{data['risk']['risk_score']}/100")
            st.write(data["risk"]["rationale"])
            st.write(f"Portfolio concentration: {data['risk']['computed_concentration_pct']}%")

        st.info("🔒 Human-in-the-loop required — trade cannot execute automatically.")

    if st.session_state.get("pending_thread"):
        st.markdown("---")
        st.markdown(f"**Awaiting human decision for thread:** `{st.session_state['pending_thread']}`")
        d1, d2 = st.columns(2)
        if d1.button("✅ Approve Trade"):
            r = requests.post(f"{API_BASE}/trade/resolve",
                               json={"thread_id": st.session_state["pending_thread"], "decision": "approve"})
            st.success(f"Final status: {r.json()['final_status']}")
        if d2.button("❌ Reject Trade"):
            r = requests.post(f"{API_BASE}/trade/resolve",
                               json={"thread_id": st.session_state["pending_thread"], "decision": "reject"})
            st.error(f"Final status: {r.json()['final_status']}")
