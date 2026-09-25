"""A mirror for exercise starters: run each task, check what it returned.

Not a test framework and not a grader. Each check asserts a *property* a correct
answer must have (a join that did not fan out, a split with no future in it),
never one exact number — the data can be regenerated and your model is allowed
to differ from ours.

    CHECKS = [
        Task("1. Plan report", task_1, check_1),
        Task("4. Foot-gun hunt", None, None, writeup=True),
    ]
    run(CHECKS)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Task:
    name: str
    solve: Callable[[], Any] | None
    check: Callable[[Any], None] | None
    writeup: bool = False


def expect(condition: bool, message: str) -> None:
    """assert, with a message written for a learner rather than a stack trace."""
    if not condition:
        raise AssertionError(message)


def run(tasks: list[Task]) -> bool:
    passed = checked = not_started = failed = 0
    for task in tasks:
        if task.writeup or task.solve is None:
            print(f"  ✎  {task.name} — a write-up; no code to check")
            continue
        checked += 1
        try:
            result = task.solve()
        except NotImplementedError:
            print(f"  ·  {task.name} — not started")
            not_started += 1
            continue
        except Exception:  # noqa: BLE001 — show the learner their own error
            print(f"  ✗  {task.name} — your code raised:")
            print("     " + traceback.format_exc(limit=2).strip().replace("\n", "\n     "))
            failed += 1
            continue
        try:
            if task.check is not None:
                task.check(result)
        except AssertionError as exc:
            print(f"  ✗  {task.name} — {exc}")
            failed += 1
            continue
        passed += 1
        print(f"  ✓  {task.name}")
    print(f"\n{passed}/{checked} checked tasks pass.")
    if failed:
        print("A ✗ line says what a right answer looks like. Stuck? Open the task's hints on the exercise page.")
    elif not_started == checked and checked:
        print("Next: open this file, fill in task_1 (delete its `raise NotImplementedError`), and run it again.")
    return passed == checked
