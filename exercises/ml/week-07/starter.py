"""Week 7 exercises — Classification: A Score, Then a Threshold.

Run from the repo root:

    python exercises/ml/week-07/starter.py

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
from pipelines.features import FEATURE_COLS, NUMERIC, build_features, make_preprocessor  # noqa: F401

df = build_features(as_of="2024-06-01")  # ~8k laptop sample, lifetime label (Week 8 fixes it)


def task_1() -> tuple[np.ndarray, np.ndarray, float]:
    """(forest test scores, test labels as an array, precision among the top 100 scores)."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """One row per cut in 0.1 … 0.9 with columns cut, flagged, precision, recall."""
    raise NotImplementedError


def task_3() -> tuple[float, float]:
    """(test AUC with every feature, test AUC with tenure_so_far removed)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    proba, y, p100 = result
    expect(len(proba) == len(y), "scores and labels should be the same test rows")
    top = np.argsort(-np.asarray(proba))[:100]
    expect(abs(p100 - np.asarray(y)[top].mean()) < 1e-9, "precision@100 = share of churners among the 100 highest scores")
    expect(p100 > np.mean(y), f"your top 100 ({p100:.2f}) should beat the base rate ({np.mean(y):.2f}) — check the sort direction")


def check_2(sweep: pd.DataFrame) -> None:
    for col in ["cut", "flagged", "precision", "recall"]:
        expect(col in sweep.columns, f"missing column {col!r}")
    sweep = sweep.sort_values("cut")
    expect(len(sweep) >= 9, "sweep 0.1, 0.2, … 0.9")
    expect(sweep["flagged"].is_monotonic_decreasing, "a higher cut can only flag fewer people")
    expect(sweep["recall"].is_monotonic_decreasing, "a higher cut can only catch fewer churners (recall falls)")


def check_3(result) -> None:
    full, without = result
    expect(0.5 < full < 1 and 0.5 < without < 1, "both AUCs should sit between a coin flip (0.5) and perfect (1.0)")
    expect(without <= full + 0.02, "removing a feature should not make the model clearly better — same split, same seed?")


CHECKS = [
    Task("0. Predict first", None, None, writeup=True),
    Task("1. Capacity budget", task_1, check_1),
    Task("2. Threshold sweep", task_2, check_2),
    Task("3. Ablation", task_3, check_3),
]

if __name__ == "__main__":
    run(CHECKS)
