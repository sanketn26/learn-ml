"""Week 15 exercises — The Pickle: A Training Script Is Not Production.

Run from the repo root:

    python exercises/ml/week-15/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS  # noqa: E402
from pipelines.split import snapshot_split  # noqa: E402

train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)


def task_1() -> dict[str, float]:
    """Test AUC for {"shuffled": ..., "signup_cut": ..., "backtest": ...} — same GBT each time."""
    raise NotImplementedError


def task_2() -> tuple[dict, float]:
    """(one predict() response, p95 latency in ms over 80 calls)."""
    raise NotImplementedError


def task_3() -> tuple[float, int]:
    """(the budget cut, how many test customers it flags) — at most 80, as close to 80 as the ties allow."""
    raise NotImplementedError


def task_4():
    """The drift figure: train vs today histograms for mrr, log_usage, tenure_so_far (three axes)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(aucs) -> None:
    expect({"shuffled", "signup_cut", "backtest"} <= set(aucs), "three splits: shuffled, signup_cut, backtest")
    expect(all(0.5 < v < 1 for v in aucs.values()), "every AUC should sit between a coin flip and perfect")
    expect(aucs["signup_cut"] < aucs["backtest"],
           "the signup-cut model scores new customers it never trained on; it should trail the backtest")


def check_2(result) -> None:
    response, p95 = result
    expect({"churn_score", "flag_for_cs", "model_version"} <= set(response), f"predict() returns {sorted(response)}")
    expect(0 <= response["churn_score"] <= 1, "churn_score is a number in [0, 1]")
    expect(p95 < 100, f"p95 {p95:.1f} ms for one row is slow — are you refitting inside the loop?")


def check_3(result) -> None:
    cut, flagged = result
    expect(0 < flagged <= 80, f"the cut flags {flagged} customers; the budget is 80")


def check_4(fig) -> None:
    expect(len(fig.axes) >= 3, "one panel per column: mrr, log_usage, tenure_so_far")


CHECKS = [
    Task("1. Time wall", task_1, check_1),
    Task("2. predict() contract", task_2, check_2),
    Task("3. Capacity, not 0.5", task_3, check_3),
    Task("4. Drift sketch", task_4, check_4),
    Task("5. One-page write-up", None, None, writeup=True),
]

if __name__ == "__main__":
    print(f"rows={len(train):,}  positives={int(y_train.sum())}  cols={FEATURE_COLS}\n")
    run(CHECKS)
