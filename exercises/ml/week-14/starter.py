"""Week 14 exercises — Neural Nets, Without the Mystique.

Run from the repo root:

    python exercises/ml/week-14/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor  # noqa: E402
from pipelines.split import snapshot_split  # noqa: E402

train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)
prep = make_preprocessor()
X_train_t = prep.fit_transform(train[FEATURE_COLS])
X_test_t = prep.transform(test[FEATURE_COLS])


def task_1() -> tuple[float, float]:
    """(logistic regression test AUC, MLPClassifier(activation="identity") test AUC)."""
    raise NotImplementedError


def task_2() -> tuple[float, float]:
    """(train AUC, test AUC) for MLPClassifier(hidden_layer_sizes=(128, 128, 128))."""
    raise NotImplementedError


def task_4() -> tuple[list[float], list[float]]:
    """(5 epoch losses WITH opt.zero_grad(), 5 epoch losses WITHOUT it), same seed."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    logreg, linear_mlp = result
    expect(logreg > 0.6 and linear_mlp > 0.6, "both should rank churners better than a coin flip")
    expect(abs(logreg - linear_mlp) < 0.05, "a net with no activation is a linear model — its AUC should land near logreg's")


def check_2(result) -> None:
    tr, te = result
    expect(tr > te, "the big net should score its own training rows higher than new ones")


def check_4(result) -> None:
    with_zg, without_zg = result
    expect(len(with_zg) == len(without_zg) == 5, "five losses each")
    expect(abs(with_zg[0] - without_zg[0]) < 1e-6, "the first step is identical (nothing has accumulated yet) — same seed?")
    expect(with_zg != without_zg, "after step 1 the runs should diverge: stale gradients pile up without zero_grad")


CHECKS = [
    Task("1. Linear MLP", task_1, check_1),
    Task("2. Too much net", task_2, check_2),
    Task("3. Decision memo", None, None, writeup=True),
    Task("4. Break the loop", task_4, check_4),
]

if __name__ == "__main__":
    print("rows", len(train), "features", FEATURE_COLS, "\n")
    run(CHECKS)
