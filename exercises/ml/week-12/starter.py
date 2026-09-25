"""Week 12 exercises — PCA: JPEG for Tables.

Run from the repo root:

    python exercises/ml/week-12/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from lib.course_data import load_customer_360  # noqa: E402

cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "avg_session", "n_support"]
sample = load_customer_360().sample(6000, random_state=0)
X = StandardScaler().fit_transform(sample[cols])


def task_1() -> np.ndarray:
    """The 2-D PCA coordinates of X (shape (6000, 2)) you colored by churn."""
    raise NotImplementedError


def task_2() -> tuple[int, list[str]]:
    """(smallest k with cumulative explained variance ≥ 0.8, the 8 user_ids worst reconstructed with k PCs)."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(xy: np.ndarray) -> None:
    expect(np.asarray(xy).shape == (len(X), 2), f"expected shape ({len(X)}, 2), got {np.asarray(xy).shape}")
    expect(np.var(xy[:, 0]) >= np.var(xy[:, 1]), "PC1 should carry at least as much variance as PC2")


def check_2(result) -> None:
    from sklearn.decomposition import PCA

    k, ids = result
    cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    want = int(np.argmax(cum >= 0.8)) + 1
    expect(k == want, f"k={k}: the first k with cumulative variance ≥ 0.8 is {want} (cumulative: {np.round(cum, 2)})")
    expect(len(ids) == 8 and set(ids) <= set(sample["user_id"]), "eight user_ids from the sample")


CHECKS = [
    Task("1. Color by churn", task_1, check_1),
    Task("2. How many components?", task_2, check_2),
    Task("3. Do not ship PC3", None, None, writeup=True),
]

if __name__ == "__main__":
    run(CHECKS)
