"""Durable state for the agent: checkpoints and the ledger, both on disk (LangGraph Week 3).

`InMemorySaver` keeps a paused refund alive only as long as the process.
A deploy, a crash, or an OOM kill forgets it — and the customer's refund
with it. `open_store(directory)` returns a SQLite checkpointer and a
SQLite ledger; a new process that opens the same directory finds the
refund still paused on the same thread_id, and the credits already paid.

Needs the framework venv (`langgraph-checkpoint-sqlite`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from capstone_agent.ledger import Ledger


def open_store(directory: str | Path):
    """(checkpointer, ledger) backed by files in `directory`. Call again after a restart."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(directory / "checkpoints.sqlite"), check_same_thread=False)
    return SqliteSaver(conn), Ledger(directory / "ledger.sqlite")


def close_store(checkpointer, ledger: Ledger) -> None:
    """What a process exit does: drop both connections. Nothing in memory survives."""
    checkpointer.conn.close()
    ledger.close()
