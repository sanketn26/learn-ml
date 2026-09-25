"""Week 0 exercises — Strong Python for AI Engineers.

Run from the repo root:

    python exercises/ml/week-00/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run
from lib.course_data import find_data_dir

DATA = find_data_dir()


def task_1() -> dict[str, float]:
    """Churn rate per plan_type, using only csv + Counter. Return {plan: rate}."""
    raise NotImplementedError


def task_2() -> type:
    """Return a CustomerFeatures dataclass with from_row(row) and to_payload() -> dict."""
    raise NotImplementedError


def task_3() -> type:
    """Return a MeanBaseline class: fit(y) -> self, predict(n) -> list, predict before fit raises."""
    raise NotImplementedError


def task_4() -> tuple:
    """Return (add_tag_buggy, add_tag): the mutable-default version and the fix."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def _first_row() -> dict:
    with open(DATA / "subscriptions.csv", newline="") as fh:
        return next(csv.DictReader(fh))


def check_1(rates: dict[str, float]) -> None:
    heads, churned = Counter(), Counter()
    with open(DATA / "subscriptions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            heads[row["plan_type"]] += 1
            churned[row["plan_type"]] += row["is_churned"] == "1"
    expect(set(rates) == set(heads), f"expected one rate per plan {sorted(heads)}, got {sorted(rates)}")
    for plan in heads:
        truth = churned[plan] / heads[plan]
        expect(abs(rates[plan] - truth) < 1e-9, f"{plan}: got {rates[plan]:.4f}, want {truth:.4f} (churned ÷ customers)")


def check_2(cls: type) -> None:
    payload = cls.from_row(_first_row()).to_payload()
    expect(isinstance(payload, dict), "to_payload() should return a dict")
    json.dumps(payload)  # raises if something is not JSON-safe
    expect(payload.get("user_id") == _first_row()["user_id"], "payload should carry the row's user_id")
    expect(isinstance(payload.get("mrr"), float), "mrr should be a float in the payload, not the CSV string")


def check_3(cls: type) -> None:
    expect(cls().fit([2, 4, 6]).predict(2) == [4.0, 4.0], "fit([2, 4, 6]).predict(2) should be [4.0, 4.0]")
    try:
        cls().predict(1)
    except Exception:  # noqa: BLE001 — any loud failure is right
        return
    raise AssertionError("predict() before fit() should raise, not return something")


def check_4(pair: tuple) -> None:
    buggy, fixed = pair
    buggy("user_1", "vip")
    expect(len(buggy("user_2", "trial")["tags"]) == 2, "the buggy version's second call should carry the first call's tag")
    fixed("user_1", "vip")
    expect(fixed("user_2", "trial")["tags"] == ["trial"], "the fixed version's second call should only have its own tag")


CHECKS = [
    Task("1. Plan report", task_1, check_1),
    Task("2. Dataclass round-trip", task_2, check_2),
    Task("3. MeanBaseline tests", task_3, check_3),
    Task("4. Foot-gun hunt", task_4, check_4),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
