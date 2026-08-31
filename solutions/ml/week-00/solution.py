"""Week 00 reference solution — Strong Python for AI Engineers.

Run from the repo root:

    python solutions/ml/week-00/solution.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir

DATA = find_data_dir()


def plan_churn_report(path: Path) -> dict[str, tuple[int, int, float]]:
    """Churn rate per plan_type using only csv + Counter."""
    n = Counter()
    churned = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            plan = row["plan_type"]
            n[plan] += 1
            if row["is_churned"] in {"1", "true", "True"}:
                churned[plan] += 1
    report = {}
    for plan in sorted(n):
        heads = n[plan]
        left = churned[plan]
        rate = (left / heads) if heads else 0.0
        report[plan] = (left, heads, rate)
    return report


@dataclass
class CustomerFeatures:
    user_id: str
    mrr: float
    tenure_days: int
    plan_type: str
    log_usage: float = 0.0

    def is_paid(self) -> bool:
        return self.plan_type != "free"

    def to_payload(self) -> dict:
        """JSON-safe body a /predict handler could accept."""
        return {
            "user_id": self.user_id,
            "mrr": self.mrr,
            "tenure_days": self.tenure_days,
            "plan_type": self.plan_type,
            "log_usage": self.log_usage,
        }


class MeanBaseline:
    """Predicts the training-set average forever. Beat this or go home."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y: Iterable[float]) -> MeanBaseline:
        values = list(y)
        if not values:
            raise ValueError("fit() needs at least one target")
        self.mean_ = sum(values) / len(values)
        return self

    def predict(self, n: int) -> list[float]:
        if self.mean_ is None:
            raise RuntimeError("call fit() before predict()")
        return [self.mean_] * n


def add_tag_buggy(user: dict, tag: str, tags: list[str] = []) -> list[str]:
    """Foot-gun: the default list is created once, at function definition."""
    tags.append(tag)
    return tags


def add_tag(user: dict, tag: str, tags: list[str] | None = None) -> list[str]:
    tags = list(tags) if tags is not None else []
    tags.append(tag)
    return tags


def main() -> None:
    print(f"data: {DATA}")

    print("\n1. Plan report (csv + Counter)")
    report = plan_churn_report(DATA / "subscriptions.csv")
    for plan, (left, heads, rate) in report.items():
        print(f"  {plan:12s}  churn={rate:.3f}  ({left}/{heads})")

    print("\n2. Dataclass round-trip")
    with (DATA / "subscriptions.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    features = CustomerFeatures(
        user_id=row["user_id"],
        mrr=float(row["mrr"]),
        tenure_days=int(row["tenure_days"]),
        plan_type=row["plan_type"],
    )
    payload = features.to_payload()
    print("  payload:", payload)
    assert isinstance(payload, dict)
    assert payload["user_id"] == features.user_id

    print("\n3. MeanBaseline tests")
    assert MeanBaseline().fit([2, 4, 6]).predict(2) == [4.0, 4.0]
    raised = False
    try:
        MeanBaseline().predict(1)
    except RuntimeError:
        raised = True
    assert raised, "predict-before-fit must raise"
    print("  fit([2,4,6]).predict(2) == [4.0, 4.0]")
    print("  predict before fit raises RuntimeError")

    print("\n4. Mutable-default foot-gun, then the fix")
    dirty_a = add_tag_buggy({"id": "u1"}, "vip")
    dirty_b = add_tag_buggy({"id": "u2"}, "vip")
    print("  buggy second call:", dirty_b, "  dirty?", dirty_b is dirty_a or len(dirty_b) > 1)
    clean_a = add_tag({"id": "u1"}, "vip")
    clean_b = add_tag({"id": "u2"}, "vip")
    print("  fixed second call:", clean_b)
    assert clean_b == ["vip"]
    assert clean_a is not clean_b


if __name__ == "__main__":
    main()
