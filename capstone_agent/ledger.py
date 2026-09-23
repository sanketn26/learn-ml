"""Billing's credit ledger — the one write the agent can make (LangGraph Weeks 4/5).

`credit(key, ...)` is idempotent on `key`: a replay returns the first
result instead of paying twice. `crash_after_next_write()` makes the next
new write land and then raise, like a process that dies after billing
said 200 but before LangGraph saved the checkpoint. On resume the node
runs again; only the key stands between the customer and a second credit.
"""

from __future__ import annotations


class ProcessDied(RuntimeError):
    """The write landed; the checkpoint did not."""


class Ledger:
    def __init__(self) -> None:
        self.credits: dict[str, dict] = {}
        self.calls = 0
        self._crash_next = False

    def crash_after_next_write(self) -> None:
        self._crash_next = True

    def credit(self, key: str, user_id: str, cents: int) -> dict:
        self.calls += 1
        if key in self.credits:
            return {**self.credits[key], "replayed": True}
        result = {"key": key, "user_id": user_id, "cents": cents, "status": "credited"}
        self.credits[key] = result
        if self._crash_next:
            self._crash_next = False
            raise ProcessDied(f"died after writing {key}")
        return {**result, "replayed": False}

    def total_cents(self, user_id: str) -> int:
        return sum(c["cents"] for c in self.credits.values() if c["user_id"] == user_id)
