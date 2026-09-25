"""Week 18 exercises — CNNs: Sliding Detectors.

Run from the repo root:

    python exercises/ml/week-18/starter.py

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

X, y = load_weekly_usage_grid()  # at-risk customers × 12 weeks before 2024-06-01; y = churn in the next 30 days
idx = np.random.default_rng(0).permutation(len(X))
cut = int(0.8 * len(X))
Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def task_1() -> dict[int, tuple[float, float]]:
    """{kernel_size: (test PR-AUC, test AUC)} for kernel_size 3 and 5, same seed and split."""
    raise NotImplementedError


def task_2() -> tuple[float, float]:
    """(test PR-AUC, test AUC) of nn.Linear(12, 1) on the same split."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def _check_pair(pr: float, auc: float, label: str) -> None:
    expect(0 <= pr <= 1 and 0 <= auc <= 1, f"{label}: PR-AUC and AUC are between 0 and 1")
    expect(auc > 0.5, f"{label}: AUC {auc:.3f} — no better than a coin; is the model training at all?")
    expect(pr > yte.mean(), f"{label}: PR-AUC {pr:.3f} should beat the base rate {yte.mean():.3f}")


def check_1(result) -> None:
    expect({3, 5} <= set(result), "kernel sizes 3 and 5")
    for k in (3, 5):
        _check_pair(*result[k], f"kernel={k}")


def check_2(result) -> None:
    _check_pair(*result, "dense")


CHECKS = [
    Task("1. Kernel size", task_1, check_1),
    Task("2. Dense baseline", task_2, check_2),
    Task("3. Draw it", None, None, writeup=True),
]

if __name__ == "__main__":
    print(f"users={len(X):,}  timesteps={X.shape[1]}  churn={y.mean():.3f}\n")
    run(CHECKS)
