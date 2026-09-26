"""Week 13 exercises — Ensembles: A Room of Reviewers.

Run from the repo root:

    python exercises/ml/week-13/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor  # noqa: E402, F401
from pipelines.split import snapshot_split  # noqa: E402

train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)


def task_1() -> pd.Series:
    """GBT feature_importances_ indexed by the fitted prep's get_feature_names_out()."""
    raise NotImplementedError


def task_2() -> dict[str, tuple[float, float]]:
    """{"shallow": (train AUC, test AUC), "deep": (train AUC, test AUC)}; deep = max_depth=8, n_estimators=80."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(imp: pd.Series) -> None:
    names = set(make_preprocessor().fit(train[FEATURE_COLS]).get_feature_names_out())
    expect(set(imp.index) == names, "index by the ENCODED names (num__mrr, cat__plan_type_free, …), not FEATURE_COLS")
    expect(abs(imp.sum() - 1) < 1e-6, "tree importances sum to 1 — is this the fitted model's feature_importances_?")


def check_2(aucs) -> None:
    (s_tr, s_te), (d_tr, d_te) = aucs["shallow"], aucs["deep"]
    expect(d_tr > s_tr, "the deep ensemble should fit its own training rows better than the shallow one")
    expect(d_tr - d_te > s_tr - s_te, "the deep ensemble's train–test gap should be wider — that gap is overfitting")


CHECKS = [
    Task("1. Feature importance", task_1, check_1),
    Task("2. Push the depth", task_2, check_2),
    Task("3. Naming quiz", None, None, writeup=True),
]

if __name__ == "__main__":
    print("rows", len(train), "positives", int(y_train.sum()), "\n")
    run(CHECKS)
