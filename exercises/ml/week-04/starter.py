"""Week 4 exercises — Charts That Change a Decision.

Run from the repo root:

    python exercises/ml/week-04/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
Charts are saved as PNGs next to where you run the script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from lib.course_data import find_data_dir, load_customer_360  # noqa: E402

DATA = find_data_dir()
df = load_customer_360(DATA)


def task_1() -> tuple[pd.Series, plt.Figure]:
    """(mean features_adopted per plan_type, the bar chart with the winner annotated)."""
    raise NotImplementedError


def task_2() -> tuple[pd.Series, plt.Figure]:
    """(churn rate per most-common region, a horizontal sorted bar chart)."""
    raise NotImplementedError


def task_3() -> plt.Figure:
    """The plan-churn bar with a y-axis from 0 and a title that is a claim the numbers support."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    adopt, fig = result
    truth = df[df["signup_date"].notna()].groupby("plan_type")["features_adopted"].mean()
    expect(set(adopt.index) == set(truth.index), "one bar per plan_type")
    expect((adopt - truth.reindex(adopt.index)).abs().max() < 1e-9, "mean features_adopted per plan, over users with a signup_date")
    expect(any(ax.texts for ax in fig.axes), "annotate the winning bar (ax.annotate / ax.text)")


def check_2(result) -> None:
    churn, fig = result
    expect(set(churn.index) - {"unknown"} <= {"NAMER", "EMEA", "APAC", "LATAM"}, f"regions look wrong: {sorted(churn.index)}")
    expect(churn.is_monotonic_increasing or churn.is_monotonic_decreasing, "sort the bars — an unsorted bar chart hides the ranking")
    expect(churn.between(0, 1).all(), "churn rate is a share between 0 and 1")


def check_3(fig) -> None:
    ax = fig.axes[0]
    expect(ax.get_ylim()[0] == 0, f"the y-axis starts at {ax.get_ylim()[0]:g}; an honest bar starts at 0")
    title = ax.get_title().strip()
    expect(title and not title.upper().startswith("TODO"), "write the title")
    expect(not title.lower().startswith(("churn by", "churn rate by", "average")),
           f"{title!r} names the chart; make it a claim (\"Free churns ~2× paid\")")


CHECKS = [
    Task("1. Adoption curve", task_1, check_1),
    Task("2. Region bars", task_2, check_2),
    Task("3. Honest title", task_3, check_3),
]

if __name__ == "__main__":
    print(f"data: {DATA}\n")
    run(CHECKS)
