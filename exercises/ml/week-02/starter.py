"""Week 2 exercises — Pandas: SQL You Already Know, in Python.

Run from the repo root:

    python exercises/ml/week-02/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run
from lib.course_data import find_data_dir

DATA = find_data_dir()
subs = pd.read_csv(DATA / "subscriptions.csv")


def task_1() -> pd.DataFrame:
    """One row per plan_type (the index) with columns users, churn_rate, arpu."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """subscriptions with one extra column, `region` — each user's most-common region. Still one row per user."""
    raise NotImplementedError


def task_3():
    """Return your validate_join(left, out, key, metric) function. It must raise when the join fanned out."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(snap: pd.DataFrame) -> None:
    expect(set(snap.index) == set(subs["plan_type"]), "index the table by plan_type — one row per plan")
    for col in ["users", "churn_rate", "arpu"]:
        expect(col in snap.columns, f"missing column {col!r}")
    expect(int(snap["users"].sum()) == len(subs), "users should add up to every customer exactly once")
    truth = subs.groupby("plan_type")["is_churned"].mean()
    expect((snap["churn_rate"] - truth.reindex(snap.index)).abs().max() < 1e-9, "churn_rate should be mean(is_churned) per plan")


def check_2(joined: pd.DataFrame) -> None:
    expect(len(joined) == len(subs), f"the join changed the row count: {len(subs):,} → {len(joined):,}. Collapse events to one row per user first")
    expect(joined["user_id"].is_unique, "user_id is duplicated after the join — the many-side was not aggregated")
    expect("region" in joined.columns, "add a `region` column")
    known = set(joined["region"].dropna())
    expect(known and known <= {"NAMER", "EMEA", "APAC", "LATAM"}, f"unexpected regions {sorted(known)}")


def check_3(validate_join) -> None:
    left = pd.DataFrame({"user_id": ["a", "b"], "mrr": [1.0, 2.0]})
    ok = left.copy()
    validate_join(left, ok, "user_id", "mrr")  # a clean join must not raise
    fanned = pd.concat([left, left])
    try:
        validate_join(left, fanned, "user_id", "mrr")
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("validate_join let a doubled table through — it should raise when out rows > 1.01 × in rows")


CHECKS = [
    Task("1. Plan snapshot", task_1, check_1),
    Task("2. Region mix", task_2, check_2),
    Task("3. Quality check", task_3, check_3),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
