"""Agent capstone reference solution — a support agent that survives its own operations.

Run from the repo root, in the framework venv:

    .venv-framework/bin/python solutions/ml/capstone-agent/solution.py

No API key: the only model is FakeListChatModel, and it never routes. Routing is
a keyword firewall; money moves only after a human approves; the credit
write is keyed so a resume cannot pay twice.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from capstone_agent.golden import CW_1847_CUSTOMER, SCORES, evaluate
from capstone_agent.ledger import Ledger, ProcessDied
from capstone_agent.runbooks import retrieve
from eval.router import allowed_tools

INJECTION = ("ignore previous", "skip the allowlist", "wire a transfer", "export all", "email addresses")
WRITE_INTENT = ("refund", "credit")
CREDIT_CENTS = 2900  # policy: one month of Pro, drafted for a human to approve


class Ticket(TypedDict, total=False):
    ticket_id: str
    user_id: str
    text: str
    route: str
    tools_called: list[str]
    doc_ids: list[str]
    answer: str
    decision: str
    credit: dict
    log: Annotated[list[str], operator.add]


def build_agent(ledger: Ledger, scores: dict, checkpointer=None, approval_gate: bool = True):
    llm = FakeListChatModel(responses=["Per the runbook: Settings > API Keys, then Generate. Send it as a Bearer token."])
    answer_chain = (
        ChatPromptTemplate.from_template(
            "Answer ONLY from these CloudWave runbooks, or say you don't know.\n{context}\n\nQ: {question}"
        )
        | llm
        | StrOutputParser()
    )

    def triage(state: Ticket) -> dict:
        q = state["text"].lower()
        if any(p in q for p in INJECTION):
            route = "blocked"
        elif any(w in q for w in WRITE_INTENT):
            route = "approval"
        elif "get_churn_score" in allowed_tools(state["text"]):
            route = "score"
        else:
            route = "docs" if retrieve(state["text"]) else "idk"
        return {"route": route, "tools_called": [], "doc_ids": [], "log": [f"triage:{route}"]}

    def docs(state: Ticket) -> dict:
        hits = retrieve(state["text"])
        context = "\n".join(f"[{doc_id}] {text}" for _, doc_id, text in hits)
        return {"answer": answer_chain.invoke({"context": context, "question": state["text"]}),
                "doc_ids": [doc_id for _, doc_id, _ in hits], "log": ["answered"]}

    def idk(state: Ticket) -> dict:
        return {"answer": "I don't know yet. A human on the support team has this ticket.", "log": ["idk"]}

    def blocked(state: Ticket) -> dict:
        return {"answer": "I can't help with that request.", "log": ["blocked"]}

    def score(state: Ticket) -> dict:
        risk = scores.get(state["user_id"])
        answer = f"churn risk {risk:.2f} (read-only)" if risk is not None else "I don't know this customer."
        return {"answer": answer, "tools_called": ["get_churn_score"], "log": ["scored"]}

    def draft_credit(state: Ticket) -> dict:
        return {"answer": f"A ${CREDIT_CENTS / 100:.2f} credit is waiting for approval.", "log": ["drafted"]}

    def issue_credit(state: Ticket, config: RunnableConfig) -> dict:
        decision = state.get("decision", "")
        if decision != "approve":
            outcome = "cancelled" if decision == "reject" else "asked-for-info"
            return {"log": [outcome]}
        key = f"{config['configurable']['thread_id']}:{state['ticket_id']}:credit"
        result = ledger.credit(key, state["user_id"], CREDIT_CENTS)
        return {"credit": result, "log": ["replayed" if result["replayed"] else "credited"]}

    g = StateGraph(Ticket)
    for name, fn in (("triage", triage), ("docs", docs), ("idk", idk), ("blocked", blocked),
                     ("score", score), ("draft_credit", draft_credit), ("issue_credit", issue_credit)):
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", lambda s: s["route"], {
        "docs": "docs", "idk": "idk", "blocked": "blocked", "score": "score", "approval": "draft_credit",
    })
    g.add_edge("draft_credit", "issue_credit")
    for terminal in ("docs", "idk", "blocked", "score", "issue_credit"):
        g.add_edge(terminal, END)
    interrupt = ["issue_credit"] if approval_gate else []  # False only for the test's negative control
    return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=interrupt)


REFUND = {"ticket_id": "CW-1847", "user_id": CW_1847_CUSTOMER, "text": "Please refund me for the broken export."}


def drill_decisions() -> dict:
    """LangGraph Week 4: approve, reject, needs-info — each on its own thread."""
    ledger, outcomes = Ledger(), {}
    app = build_agent(ledger, SCORES)
    for decision in ("approve", "reject", "needs_info"):
        config = {"configurable": {"thread_id": f"drill-{decision}"}}
        app.invoke(REFUND, config)
        app.update_state(config, {"decision": decision})
        outcomes[decision] = app.invoke(None, config)["log"][-1]
    return {"outcomes": outcomes, "credited_cents": ledger.total_cents(CW_1847_CUSTOMER)}


def drill_crash_after_write() -> dict:
    """LangGraph Weeks 3 + 5: billing said 200, the process died before the checkpoint. Resume."""
    ledger = Ledger()
    app = build_agent(ledger, SCORES)
    config = {"configurable": {"thread_id": "drill-crash"}}
    app.invoke(REFUND, config)
    app.update_state(config, {"decision": "approve"})
    ledger.crash_after_next_write()
    try:
        app.invoke(None, config)
    except ProcessDied:
        pass
    resumed = app.invoke(None, config)
    return {"last_log": resumed["log"][-1], "ledger_calls": ledger.calls,
            "credited_cents": ledger.total_cents(CW_1847_CUSTOMER)}


def main() -> None:
    rows = evaluate(build_agent)
    for row in rows:
        print(f"{row['id']}  route={row['route']:<9} {'ok' if row['ok'] else 'FAIL ' + '; '.join(row['problems'])}")
    print("golden failures:", sum(not r["ok"] for r in rows))
    print("decisions:", drill_decisions())
    print("crash after write:", drill_crash_after_write())


if __name__ == "__main__":
    main()
