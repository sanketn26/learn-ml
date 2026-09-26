"""Week 11 exercises — Rank a List.

Run from the repo root:

    python exercises/ml/week-11/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.checks import Task, expect, run  # noqa: E402
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor  # noqa: E402
from pipelines.split import snapshot_split  # noqa: E402

# Backtest: learn on the snapshot 30 days earlier, rank today's customers.
train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)
model = Pipeline([
    ("prep", make_preprocessor()),
    ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
]).fit(train[FEATURE_COLS], y_train)
scores = model.predict_proba(test[FEATURE_COLS])[:, 1]
y_test = y_test.to_numpy()


def precision_at_k(y, s, k=80) -> float:
    return float(np.asarray(y)[np.argsort(-np.asarray(s))[:k]].mean())


def task_1() -> dict[str, tuple[float, float]]:
    """{ranker name: (precision@80, recall@80)} for "gbt", "n_support", "-log_usage", "random"."""
    raise NotImplementedError


def task_2() -> dict[int, float]:
    """{k: GBT precision@k} for k in 20, 80, 200."""
    raise NotImplementedError


def task_5() -> tuple[float, float, float, float]:
    """(p@80 CI low, p@80 CI high, gap-vs-(-log_usage) CI low, gap CI high) from a 1,000-draw paired bootstrap."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(table) -> None:
    expect({"gbt", "n_support", "-log_usage", "random"} <= set(table), "four rankers, named as in the task")
    positives = y_test.sum()
    p, r = table["gbt"]
    expect(abs(p - precision_at_k(y_test, scores)) < 1e-9, "gbt precision@80 should use this backtest's scores")
    expect(abs(r - p * 80 / positives) < 1e-9, "recall@80 = hits in the top 80 ÷ all churners in the test set")
    expect(table["gbt"][0] > table["random"][0], "the GBT should beat a random 80 — check the sort direction")


def check_2(by_k) -> None:
    expect({20, 80, 200} <= set(by_k), "k = 20, 80, and 200")
    for k in (20, 80, 200):
        expect(abs(by_k[k] - precision_at_k(y_test, scores, k)) < 1e-9, f"precision@{k} looks off")


def check_5(result) -> None:
    lo, hi, g_lo, g_hi = result
    point = precision_at_k(y_test, scores)
    expect(lo <= point <= hi, f"the interval [{lo:.3f}, {hi:.3f}] should contain the point estimate {point:.3f}")
    expect(hi - lo > 0.05, "an interval narrower than 5 points on 80 calls is suspicious — did you resample rows with replacement?")
    expect(g_lo < g_hi, "the gap interval is (low, high)")


CHECKS = [
    Task("1. Four rankers", task_1, check_1),
    Task("2. Capacity", task_2, check_2),
    Task("3. Pre-register", None, None, writeup=True),
    Task("4. Causal trap", None, None, writeup=True),
    Task("5. Put an interval on it", task_5, check_5),
]

if __name__ == "__main__":
    print("positives in test", int(y_test.sum()), "of", len(y_test), "\n")
    run(CHECKS)
