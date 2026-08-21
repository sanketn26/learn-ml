"""Week 9 exercises — Week 9 — Regression: Predict a Number, Not a Category.

Run from the repo root:

    python exercises/ml/week-09/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import AS_OF_DEFAULT, build_features


def main() -> None:
    df = build_features(as_of=AS_OF_DEFAULT, n=8000, at_risk_only=True)
    print("as-of rows", len(df), "cols", ["tenure_so_far", "total_usage", "plan_type"])
    print("TODO 1: forest on log1p(total_usage), expm1 the predictions")
    print("TODO 2: MAE sliced by plan_type")
    print("TODO 3: fake_clv = mrr * (tenure_so_far / 30) — then delete it")


if __name__ == "__main__":
    main()
