"""Week 6 exercises — Features Are the Model’s API.

Run from the repo root:

    python exercises/ml/week-06/starter.py

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
from pipelines.features import NUMERIC, build_features

df = build_features(as_of="2024-06-01")  # ~8k laptop sample


def task_1() -> tuple[float, float, int, int]:
    """(mrr mean the all-rows scaler used, mrr mean the train-only scaler used, n all rows, n train rows)."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """Churn rate (is_churned mean) grouped by has_usage = total_usage > 0. Index: False/True."""
    raise NotImplementedError


def task_3():
    """Return assert_score_payload(payload): raises on missing keys, extra keys, or wrong types."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    mean_all, mean_train, n_all, n_train = result
    expect(n_all == len(df), f"the all-rows scaler should see all {len(df):,} rows")
    expect(n_train < n_all, "the train-only scaler should see fewer rows than the all-rows one")
    expect(abs(mean_all - df["mrr"].mean()) < 1e-9, "the all-rows mean is just df['mrr'].mean()")
    expect(mean_all != mean_train, "the two means should differ — that difference is the leak, however small")


def check_2(table) -> None:
    rates = table["mean"] if isinstance(table, pd.DataFrame) else table
    expect(set(rates.index) == {False, True}, "group by has_usage: one row for False, one for True")
    has = df["total_usage"] > 0
    for flag in (False, True):
        expect(abs(rates[flag] - df.loc[has == flag, "is_churned"].mean()) < 1e-9, f"churn rate for has_usage={flag} looks off")


def check_3(assert_score_payload) -> None:
    good = {"mrr": 49.0, "tenure_so_far": 120, "log_usage": 3.2, "plan_type": "pro"}
    assert_score_payload(good)
    bad = {
        "a missing key": {k: v for k, v in good.items() if k != "mrr"},
        "an extra key (churn_date)": {**good, "churn_date": "2024-07-01"},
        "a string where a number belongs": {**good, "mrr": "49"},
    }
    for label, payload in bad.items():
        try:
            assert_score_payload(payload)
        except Exception:  # noqa: BLE001
            continue
        raise AssertionError(f"a payload with {label} got through")


CHECKS = [
    Task("0. Predict first", None, None, writeup=True),
    Task("1. Two scaler fits", task_1, check_1),
    Task("2. Missingness flag", task_2, check_2),
    Task("3. Contract test", task_3, check_3),
]

if __name__ == "__main__":
    run(CHECKS)
