"""Week 13 exercises — Week 13 — Ensembles: A Room of Reviewers.

Run from the repo root:

    python exercises/ml/week-13/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_eventual_churn


def main() -> None:
    df = build_features(as_of=AS_OF_DEFAULT, n=None, at_risk_only=True)
    y = label_eventual_churn(df, AS_OF_DEFAULT)
    df, y = drop_unlabelled(df, y)
    print("rows", len(df), "positives", int(y.sum()), "features", FEATURE_COLS)
    print("preprocessor", make_preprocessor())
    print("TODO 1: GBT feature_importances_ next to get_feature_names_out()")
    print("TODO 2: max_depth=8, n_estimators=80 — train AUC vs test AUC")


if __name__ == "__main__":
    main()
