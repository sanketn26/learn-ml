"""Week 19 exercises — RNNs: A Clipboard That Walks the Sequence.

Run from the repo root:

    python exercises/ml/week-19/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from lib.course_data import load_weekly_usage_grid  # noqa: E402

X, y = load_weekly_usage_grid(random_state=1)
idx = np.random.default_rng(1).permutation(len(X))
cut = int(0.8 * len(X))
Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def task_1() -> dict[str, float]:
    """{"last": test AUC reading out[:, -1], "mean": test AUC reading out.mean(dim=1)} — same GRU, same seed."""
    raise NotImplementedError


def task_2() -> tuple[float, float]:
    """(test AUC on forward weeks, test AUC after flipping both train and test sequences)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(aucs) -> None:
    expect({"last", "mean"} <= set(aucs), "one AUC for each pooling: 'last' and 'mean'")
    expect(all(a > 0.5 for a in aucs.values()), f"both should beat a coin flip: {aucs}")


def check_2(result) -> None:
    forward, reversed_ = result
    expect(forward > 0.5 and reversed_ > 0.5, "both runs should beat a coin flip — the total usage is still there when you flip")
    expect(forward != reversed_, "identical AUCs: did you actually flip the sequences (X[:, ::-1])?")


CHECKS = [
    Task("1. Use the mean hidden state", task_1, check_1),
    Task("2. Reverse the weeks", task_2, check_2),
    Task("3. One-sentence LSTM", None, None, writeup=True),
]

if __name__ == "__main__":
    print(Xtr.shape, "churn", float(ytr.mean()), "\n")
    run(CHECKS)
