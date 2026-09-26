"""Billing's credit ledger — the one write the agent can make (LangGraph Weeks 4/5).

`credit(key, ...)` is idempotent on `key`: a replay returns the first
result instead of paying twice. `crash_after_next_write()` makes the next
new write land and then raise, like a process that dies after billing
said 200 but before LangGraph saved the checkpoint. On resume the node
runs again; only the key stands between the customer and a second credit.

It is a SQLite table, not a dict, because billing outlives your process.
`Ledger()` is an in-memory database (tests, golden tickets); `Ledger(path)`
is a file that a restarted agent reopens and finds every credit in.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class ProcessDied(RuntimeError):
    """The write landed; the checkpoint did not."""


class Ledger:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS credits (key TEXT PRIMARY KEY, user_id TEXT, cents INTEGER)")
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (n INTEGER)")
        self._db.commit()
        self._crash_next = False

    @property
    def credits(self) -> dict[str, dict]:
        rows = self._db.execute("SELECT key, user_id, cents FROM credits").fetchall()
        return {k: {"key": k, "user_id": u, "cents": c, "status": "credited"} for k, u, c in rows}

    @property
    def calls(self) -> int:
        """Billing calls ever made against this ledger — including by a process that has since died."""
        return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def crash_after_next_write(self) -> None:
        self._crash_next = True

    def credit(self, key: str, user_id: str, cents: int) -> dict:
        self._db.execute("INSERT INTO calls VALUES (1)")
        self._db.commit()
        if key in self.credits:
            return {**self.credits[key], "replayed": True}
        self._db.execute("INSERT INTO credits VALUES (?, ?, ?)", (key, user_id, cents))
        self._db.commit()  # the write has landed, whatever happens next
        if self._crash_next:
            self._crash_next = False
            raise ProcessDied(f"died after writing {key}")
        return {**self.credits[key], "replayed": False}

    def total_cents(self, user_id: str) -> int:
        row = self._db.execute("SELECT COALESCE(SUM(cents), 0) FROM credits WHERE user_id = ?", (user_id,)).fetchone()
        return row[0]

    def close(self) -> None:
        self._db.close()
