"""Week 9 exercises — Regression: Predict a Number, Not a Category.

Run from the repo root:

    python exercises/ml/week-09/starter.py

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
from pipelines.features import AS_OF_DEFAULT, build_features

usage = pd.read_csv(find_data_dir() / "feature_usage.csv", usecols=["user_id", "usage_count", "date"], parse_dates=["date"])


def next_30d_usage(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """The lesson's label: usage in (as_of, as_of + 30 days]."""
    window = usage[(usage["date"] > as_of) & (usage["date"] <= as_of + pd.Timedelta(days=30))]
    return frame["user_id"].map(window.groupby("user_id")["usage_count"].sum()).fillna(0)


train_as_of = AS_OF_DEFAULT - pd.Timedelta(days=30)
train = build_features(as_of=train_as_of, n=8000)
test = build_features(as_of=AS_OF_DEFAULT, n=8000, random_state=7)
y_train, y_test = next_30d_usage(train, train_as_of), next_30d_usage(test, AS_OF_DEFAULT)


def task_1() -> tuple[float, float, np.ndarray]:
    """(test MAE with a raw target, test MAE with a log1p target back-transformed, the log-target predictions)."""
    raise NotImplementedError


def task_2() -> pd.DataFrame:
    """MAE per plan_type (index) on the test snapshot, column `mae`."""
    raise NotImplementedError


def task_3() -> float:
    """Test R² of a linear model predicting fake_clv = mrr * tenure_so_far / 30 from mrr and tenure_so_far."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    mae_raw, mae_log, log_pred = result
    baseline = (y_test - y_train.mean()).abs().mean()
    expect(len(log_pred) == len(y_test), "one prediction per test customer")
    expect(np.asarray(log_pred).min() >= 0, "expm1 your log-target predictions back to usage units (no negatives)")
    expect(mae_raw < baseline and mae_log < baseline, f"both models should beat predicting the mean (MAE {baseline:.1f})")


def check_2(slices: pd.DataFrame) -> None:
    expect({"free", "enterprise"} <= set(slices.index), "include at least free and enterprise")
    expect("mae" in slices.columns and (slices["mae"] >= 0).all(), "a non-negative `mae` column per plan")


def check_3(r2: float) -> None:
    expect(r2 > 0.5, f"R² {r2:.2f}: a target built from the two features should be easy to fit — that is the point")
    expect(r2 < 0.999, "a *linear* model cannot fit a product exactly; an R² of 1.0 means you fed it the target")


CHECKS = [
    Task("1. Log target", task_1, check_1),
    Task("2. Residual slices", task_2, check_2),
    Task("3. An alternative target", task_3, check_3),
]

if __name__ == "__main__":
    run(CHECKS)
