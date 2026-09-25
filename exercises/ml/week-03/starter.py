"""Week 3 exercises — SQL Is the Source of Truth.

Run from the repo root:

    python exercises/ml/week-03/starter.py

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
AS_OF = pd.Timestamp("2024-06-01")


def task_1() -> tuple[int, int]:
    """(usage rows with date <= 2024-06-01, all usage rows) — DuckDB or Pandas."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """The as_of=2024-06-01 frame (build_features(as_of=..., n=None) or the lesson's SQL), grain-tested."""
    raise NotImplementedError


def task_3() -> pd.DataFrame:
    """Five users with user_id, tenure_days, tenure_so_far — at least one of them churned after as_of."""
    raise NotImplementedError


def task_4() -> pd.Timestamp:
    """The latest legal as_of in this universe (the last day both usage and events cover)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def _usage_dates() -> pd.Series:
    return pd.read_csv(DATA / "feature_usage.csv", usecols=["date"], parse_dates=["date"])["date"]


def check_1(counts: tuple[int, int]) -> None:
    cut, total = counts
    dates = _usage_dates()
    expect(total == len(dates), f"all rows: got {total:,}, the file has {len(dates):,}")
    expect(cut == int((dates <= AS_OF).sum()), "the as-of count should include rows dated exactly 2024-06-01")


def check_2(frame: pd.DataFrame) -> None:
    subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["signup_date", "churn_date"])
    at_risk = int(((subs["signup_date"] <= AS_OF) & ~(subs["churn_date"] <= AS_OF)).sum())
    expect(frame["user_id"].is_unique, "user_id repeats — the grain is not one row per customer")
    expect(len(frame) == at_risk, f"{len(frame):,} rows vs {at_risk:,} customers at risk on {AS_OF.date()} — sampled (n=8000?) or not filtered?")
    expect((pd.to_datetime(frame["signup_date"]) <= AS_OF).all(), "someone in the frame signed up after as_of")


def check_3(five: pd.DataFrame) -> None:
    for col in ["user_id", "tenure_days", "tenure_so_far"]:
        expect(col in five.columns, f"show the {col!r} column")
    expect(len(five) == 5, "five users, please")
    expect((five["tenure_days"] != five["tenure_so_far"]).any(),
           "pick users where the two disagree — that disagreement is the lesson")


def check_4(latest: pd.Timestamp) -> None:
    usage_max = _usage_dates().max()
    events_max = pd.read_csv(DATA / "user_events.csv", usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"].max()
    want = min(usage_max, events_max.normalize())
    expect(pd.Timestamp(latest).normalize() == want, f"got {pd.Timestamp(latest).date()}, want {want.date()}: the earlier of the two tables' last days")


CHECKS = [
    Task("1. Date bound", task_1, check_1),
    Task("2. Grain test", task_2, check_2),
    Task("3. tenure_days vs tenure_so_far", task_3, check_3),
    Task("4. Freshness", task_4, check_4),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
