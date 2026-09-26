"""Week 20 exercises — Transformers: Everything Looks at Everything.

Run from the repo root:

    python exercises/ml/week-20/starter.py

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

from lib.checks import Task, expect, run  # noqa: E402
from lib.course_data import find_data_dir  # noqa: E402

fb = pd.read_json(find_data_dir() / "feedback.json", lines=True).sample(4000, random_state=0)
y = (fb["category"].str.lower() == "praise").to_numpy(dtype=np.int64)


def task_1() -> dict[str, float]:
    """{"with_pos": test accuracy, "without_pos": test accuracy} for the TinyTransformer, same seed and split."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """The 3×3 attention weights for tokens login / failed / again (rows = who is looking)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(acc) -> None:
    expect({"with_pos", "without_pos"} <= set(acc), "two runs: with_pos and without_pos")
    majority = 1 - y.mean()
    for key, value in acc.items():
        expect(value >= majority - 0.02, f"{key}: accuracy {value:.3f} is below 'always say not-praise' ({majority:.3f}) — train longer?")


def check_2(weights: pd.DataFrame) -> None:
    expect(weights.shape == (3, 3), "one row and one column per token")
    expect(np.allclose(weights.to_numpy().sum(axis=1), 1.0), "each row is a softmax: it should sum to 1")
    expect((weights.to_numpy() >= 0).all(), "attention weights are never negative")


CHECKS = [
    Task("1. Remove positions", task_1, check_1),
    Task("2. Attention map", task_2, check_2),
    Task("3. Architecture memo", None, None, writeup=True),
]

if __name__ == "__main__":
    print("comments", len(fb), "praise rate", round(float(y.mean()), 3), "\n")
    run(CHECKS)
