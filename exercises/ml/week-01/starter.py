"""Week 1 exercises — NumPy: Fast Math on Whole Columns.

Run from the repo root:

    python exercises/ml/week-01/starter.py

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
from lib.course_data import find_data_dir

DATA = find_data_dir()
usage = pd.read_csv(DATA / "feature_usage.csv")


def task_1() -> dict[str, float]:
    """Per-feature total usage_count → NumPy array → {"mean": ..., "median": ..., "p90": ...}."""
    raise NotImplementedError


def task_2() -> tuple[pd.DataFrame, np.ndarray]:
    """(pivot, normalized): users × features for a sample of users, each row divided by its own mean."""
    raise NotImplementedError


def task_3() -> tuple[list[str], float]:
    """(whale_user_ids, share_of_all_usage) for users at or above the 99th percentile of total usage."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(stats: dict[str, float]) -> None:
    totals = usage.groupby("feature_name")["usage_count"].sum().to_numpy(dtype=float)
    for key, want in [("mean", totals.mean()), ("median", np.median(totals)), ("p90", np.quantile(totals, 0.9))]:
        expect(key in stats, f"return a dict with a {key!r} key")
        expect(abs(stats[key] - want) < 1e-6 * max(want, 1),
               f"{key}: got {stats[key]:,.1f}; compute it over the {len(totals)} per-feature totals, not raw rows")


def check_2(result: tuple[pd.DataFrame, np.ndarray]) -> None:
    pivot, normalized = result
    expect(normalized.shape == pivot.shape, f"normalized {normalized.shape} should keep the pivot's shape {pivot.shape}")
    expect(np.isfinite(normalized).all(), "normalized has NaN/inf — guard rows whose mean is 0")
    busy = pivot.to_numpy(dtype=float).mean(axis=1) > 0
    expect(np.allclose(normalized[busy].mean(axis=1), 1.0),
           "each non-empty row should average 1.0 after dividing by its own mean (shape (users, 1), not (features,))")


def check_3(result: tuple[list[str], float]) -> None:
    ids, share = result
    per_user = usage.groupby("user_id")["usage_count"].sum().astype(float)
    cut = np.quantile(per_user.to_numpy(), 0.99)
    want = set(per_user[per_user >= cut].index)
    expect(set(ids) == want, f"expected the {len(want)} users at or above the 99th percentile ({cut:,.0f}), got {len(set(ids))}")
    truth = per_user.loc[list(want)].sum() / per_user.sum()
    expect(abs(share - truth) < 1e-9, f"share: got {share:.3f}, want {truth:.3f} (whale usage ÷ all usage)")


CHECKS = [
    Task("1. Feature ranking", task_1, check_1),
    Task("2. Broadcasting on a real pivot", task_2, check_2),
    Task("3. Whale hunt", task_3, check_3),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
