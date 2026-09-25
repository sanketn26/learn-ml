"""Week 5 exercises — “Is This Real, or Just Noise?”

Run from the repo root:

    python exercises/ml/week-05/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run
from lib.course_data import find_data_dir

DATA = find_data_dir()
subs = pd.read_csv(DATA / "subscriptions.csv")


def task_1() -> tuple[pd.DataFrame, float]:
    """(plan × churned crosstab for PAID plans only, chi-squared p-value). Write your prediction first."""
    raise NotImplementedError


def task_2() -> tuple[float, float, float]:
    """(mean sentiment for bug, mean sentiment for praise, Welch t-test p-value)."""
    raise NotImplementedError


def task_3() -> dict[int, float]:
    """{n: share of simulated 16%-vs-20% experiments with p < 0.05} for n in 100, 400, 1000 per group."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    table, p = result
    expect("free" not in set(table.index), "drop the free plan first")
    expect(set(table.index) == {"starter", "pro", "enterprise"}, f"rows should be the three paid plans, got {sorted(table.index)}")
    want = chi2_contingency(table)[1]
    expect(abs(p - want) < 1e-12 or (p < 1e-300 and want < 1e-300), "p should come from chi2_contingency on that table")


def check_2(result) -> None:
    bug_mean, praise_mean, p = result
    fb = pd.read_json(DATA / "feedback.json", lines=True)
    bug = fb.loc[fb["category"] == "bug", "sentiment_score"]
    praise = fb.loc[fb["category"] == "praise", "sentiment_score"]
    expect(abs(bug_mean - bug.mean()) < 1e-9 and abs(praise_mean - praise.mean()) < 1e-9, "means per category look off")
    want = ttest_ind(bug, praise, equal_var=False).pvalue
    expect(abs(p - want) <= 1e-12 + 1e-6 * want, "use Welch's t-test (equal_var=False) — the two groups have different spreads")


def check_3(power: dict[int, float]) -> None:
    expect(set(power) >= {100, 400, 1000}, "simulate n = 100, 400, and 1000")
    expect(all(0 <= power[n] <= 1 for n in power), "each value is a share of runs, between 0 and 1")
    expect(power[100] < power[400] < power[1000], "more customers should detect the same gap more often")
    expect(0.45 < power[1000] < 0.8, f"at n=1000 a 4-point gap is caught roughly 60% of the time; you got {power[1000]:.0%}")
    expect(power[100] < 0.3, f"at n=100 it is usually missed; you got {power[100]:.0%}")


CHECKS = [
    Task("1. Paid-only chi-squared", task_1, check_1),
    Task("2. Sentiment", task_2, check_2),
    Task("3. Sample size gut check", task_3, check_3),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
