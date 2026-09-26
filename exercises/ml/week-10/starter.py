"""Week 10 exercises — Clustering: Sorting Without Labels.

Run from the repo root:

    python exercises/ml/week-10/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run
from lib.course_data import load_customer_360

df = load_customer_360()
cols = ["mrr", "total_usage"]


def task_1() -> tuple[np.ndarray, np.ndarray]:
    """(K=4 cluster ids on raw mrr + total_usage, K=4 cluster ids after StandardScaler). One id per row of df."""
    raise NotImplementedError


def task_3() -> pd.Series:
    """Churn rate per cluster (index = cluster id) for your scaled clustering."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    raw, scaled = result
    for name, ids in [("raw", raw), ("scaled", scaled)]:
        expect(len(ids) == len(df), f"{name}: one cluster id per customer")
        expect(len(set(np.asarray(ids))) == 4, f"{name}: K=4 means four clusters")
    expect(not np.array_equal(pd.factorize(raw)[0], pd.factorize(scaled)[0]),
           "raw and scaled clusters are identical — did you actually scale the second run?")


def check_3(churn: pd.Series) -> None:
    expect(len(churn) == 4, "one churn rate per cluster")
    expect(churn.between(0, 1).all(), "churn rate is a share between 0 and 1")
    expect(churn.max() - churn.min() > 0.02, "the clusters should differ on churn at least a little — or there is nothing to peek at")


CHECKS = [
    Task("0. Predict first", None, None, writeup=True),
    Task("1. Raw, then scaled", task_1, check_1),
    Task("2. Name the personas", None, None, writeup=True),
    Task("3. Peek, don't train", task_3, check_3),
]

if __name__ == "__main__":
    run(CHECKS)
