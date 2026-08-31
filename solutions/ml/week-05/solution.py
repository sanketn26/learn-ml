"""Week 05 reference solution — signal vs noise.

Run from the repo root:

    python solutions/ml/week-05/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()


def simulate_power(n: int, p1: float = 0.16, p2: float = 0.20, trials: int = 400, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    hits = 0
    for _ in range(trials):
        a = rng.binomial(1, p1, size=n)
        b = rng.binomial(1, p2, size=n)
        table = np.array([[a.sum(), n - a.sum()], [b.sum(), n - b.sum()]])
        _, p, _, _ = chi2_contingency(table, correction=False)
        hits += int(p < 0.05)
    return hits / trials


def main() -> None:
    df = load_customer_360()
    print("Predict before the paid-only test: dropping free should *weaken* the plan×churn association,")
    print("because most of the original table's signal is free vs paid, not starter vs enterprise.")

    print("\n1. Paid-only chi-squared")
    paid = df[df["plan_type"] != "free"]
    table = pd.crosstab(paid["plan_type"], paid["is_churned"])
    chi2, p, dof, _ = chi2_contingency(table)
    print(table.to_string())
    print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.4g}")
    print("  still different?" , "yes" if p < 0.05 else "not at α=0.05")

    print("\n2. Sentiment: bug vs praise (two groups, a number → t-test)")
    feedback = pd.read_json(DATA / "feedback.json", lines=True)
    bug = feedback.loc[feedback["category"] == "bug", "sentiment_score"].dropna()
    praise = feedback.loc[feedback["category"] == "praise", "sentiment_score"].dropna()
    t_stat, t_p = ttest_ind(bug, praise, equal_var=False)
    print(f"  bug n={len(bug)} mean={bug.mean():.3f}  praise n={len(praise)} mean={praise.mean():.3f}")
    print(f"  Welch t={t_stat:.2f}  p={t_p:.4g}")
    print("  look at the overlap, not just p — a histogram would show two mounds, not two spikes")

    print("\n3. Sample-size gut check (16% vs 20%, equal n)")
    for n in (100, 400, 1000):
        share = simulate_power(n)
        print(f"  n={n:4d}  fraction of sims with p<0.05: {share:.2f}")


if __name__ == "__main__":
    main()
