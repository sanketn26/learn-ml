"""Week 8 exercises — Labels Lie.

Run from the repo root:

    python exercises/ml/week-08/starter.py

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
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor  # noqa: F401
from pipelines.labels import drop_unlabelled, label_churn_in_horizon

as_of = AS_OF_DEFAULT
df = build_features(as_of=as_of, n=None)


def task_1() -> tuple[float, float]:
    """(horizon-30 churn rate, lifetime is_churned rate) on the same at-risk rows."""
    raise NotImplementedError


def task_2() -> pd.Series:
    """label_churn_in_horizon with observation_end = as_of + 10 days, horizon 30."""
    raise NotImplementedError


def task_3() -> dict[str, float]:
    """GBT on the 30-day label: {"roc_auc", "pr_auc", "dummy_pr_auc", "precision_at_80"} on the test split."""
    raise NotImplementedError


def task_4() -> Exception:
    """Return the exception validate() raises for a legal payload plus churn_date."""
    raise NotImplementedError


def task_5() -> tuple[np.ndarray, np.ndarray]:
    """(fraction positive per bin, mean predicted per bin) from calibration_curve on your task 3 model."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    horizon, lifetime = result
    labelled, y = drop_unlabelled(df, label_churn_in_horizon(df, as_of))
    expect(abs(horizon - y.mean()) < 1e-9, f"horizon rate: got {horizon:.4f}, want {y.mean():.4f}")
    expect(abs(lifetime - labelled["is_churned"].mean()) < 1e-9, "lifetime rate should be is_churned on the SAME rows")
    expect(horizon < lifetime, "the 30-day rate should be smaller than the lifetime one — it asks a narrower question")


def check_2(short: pd.Series) -> None:
    expect(len(short) == len(df), "one label per row of df (NaN included)")
    expect(short.isna().sum() > 0.9 * len(df), "almost everyone should be NaN: nobody has been watched for 30 days yet")
    expect(not (short == 0).any(), "no one can be a known 0 yet — the 30-day window is not over for anyone")
    expect((short == 1).sum() > 0, "cancels already seen in the first 10 days stay 1")


def check_3(m: dict[str, float]) -> None:
    for key in ["roc_auc", "pr_auc", "dummy_pr_auc", "precision_at_80"]:
        expect(key in m, f"missing {key!r}")
    expect(m["roc_auc"] > 0.6, f"ROC-AUC {m['roc_auc']:.3f} is close to a coin flip — did you train on the 30-day label?")
    expect(m["pr_auc"] > m["dummy_pr_auc"], "PR-AUC should beat the dummy's")
    expect(m["dummy_pr_auc"] < 0.05, "the dummy PR-AUC is about the base rate (~2%); check which label you used")


def check_4(exc) -> None:
    expect(isinstance(exc, ValueError), f"validate() should raise ValueError, got {type(exc).__name__}")
    expect("churn_date" in str(exc), "the error should name the forbidden key")


def check_5(result) -> None:
    frac_pos, mean_pred = result
    expect(len(frac_pos) == len(mean_pred) and len(frac_pos) >= 3, "one point per bin, a few bins")
    expect(((0 <= np.asarray(frac_pos)) & (np.asarray(frac_pos) <= 1)).all(), "observed rates are shares between 0 and 1")


CHECKS = [
    Task("0. Predict first", None, None, writeup=True),
    Task("1. Two rates", task_1, check_1),
    Task("2. Censoring", task_2, check_2),
    Task("3. PR vs ROC", task_3, check_3),
    Task("4. Forbidden", task_4, check_4),
    Task("5. Calibration glance", task_5, check_5),
]

if __name__ == "__main__":
    run(CHECKS)
