"""Agent capstone — a CloudWave support agent that survives its own operations.

Run from the repo root, in the framework venv (make setup-frameworks):

    .venv-framework/bin/python exercises/ml/capstone-agent/starter.py

Build `build_agent`. The runner scores it against the golden tickets, then
runs the approval, crash, and restart drills. No API key: the model is an extractive
stand-in that can only repeat the runbook it was handed.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
from langchain_core.runnables import RunnableConfig  # noqa: F401
from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
from langgraph.graph import END, START, StateGraph  # noqa: F401

from capstone_agent.golden import CW_1847_CUSTOMER, SCORES, evaluate
from capstone_agent.ledger import Ledger, ProcessDied
from capstone_agent.model import extractive_model  # noqa: F401
from capstone_agent.storage import close_store, open_store  # noqa: F401
from capstone_agent.runbooks import retrieve  # noqa: F401
from eval.router import allowed_tools  # noqa: F401

CREDIT_CENTS = 2900


class Ticket(TypedDict, total=False):
    ticket_id: str
    user_id: str
    text: str
    route: str               # docs | idk | score | blocked | approval
    tools_called: list[str]
    doc_ids: list[str]
    answer: str
    decision: str            # set by a human: approve | reject | needs_info
    credit: dict
    log: Annotated[list[str], operator.add]


def build_agent(ledger: Ledger, scores: dict, checkpointer=None):
    """Return a compiled graph: checkpointer (InMemorySaver if None) + interrupt_before=["issue_credit"].

    TODO 1: triage — injection → blocked, refund/credit → approval, churn → score, else docs or idk
    TODO 2: docs (retrieve + extractive_model chain) and idk
    TODO 3: draft_credit → issue_credit, paused for a human
    TODO 5: issue_credit writes through ledger.credit with a key that survives a resume
    TODO 6: pass `checkpointer` through, so open_store's SqliteSaver makes the pause survive a restart
    """
    raise NotImplementedError("build the graph (tasks 1–3)")


REFUND = {"ticket_id": "CW-1847", "user_id": CW_1847_CUSTOMER, "text": "Please refund me for the broken export."}


def main() -> None:
    try:
        rows = evaluate(build_agent)
    except NotImplementedError as todo:
        print(f"next: {todo}")
        return
    for row in rows:
        print(f"{row['id']}  route={row['route']}  {'ok' if row['ok'] else row['problems']}")
    print("golden failures:", sum(not r["ok"] for r in rows))

    ledger = Ledger()
    app = build_agent(ledger, SCORES)
    config = {"configurable": {"thread_id": "drill-crash"}}
    app.invoke(REFUND, config)
    app.update_state(config, {"decision": "approve"})
    ledger.crash_after_next_write()
    try:
        app.invoke(None, config)
    except ProcessDied as exc:
        print("crashed:", exc)
    app.invoke(None, config)
    print(f"credited {ledger.total_cents(CW_1847_CUSTOMER)} cents over {ledger.calls} billing calls")

    import tempfile

    with tempfile.TemporaryDirectory() as store_dir:
        config = {"configurable": {"thread_id": "drill-restart"}}
        saver, ledger = open_store(store_dir)
        build_agent(ledger, SCORES, checkpointer=saver).invoke(REFUND, config)
        close_store(saver, ledger)  # the deploy
        saver, ledger = open_store(store_dir)
        waiting = build_agent(ledger, SCORES, checkpointer=saver).get_state(config).next
        close_store(saver, ledger)
        print("after a restart the refund is waiting at:", waiting, "(task 6: expect ('issue_credit',))")


if __name__ == "__main__":
    main()
