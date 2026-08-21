"""Week 8 — horizon labels vs lifetime flag.

    python exercises/ml/week-08/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon


def main() -> None:
    df = build_features(as_of=AS_OF_DEFAULT, n=None)
    y = label_churn_in_horizon(df, AS_OF_DEFAULT)
    labelled, y = drop_unlabelled(df, y)
    print("horizon rate", float(y.mean()))
    print("lifetime rate", float(labelled["is_churned"].mean()))
    print("TODO 3: GBT via make_preprocessor() (plan_type is a string) — ROC vs PR vs precision@80")
    print("prep", make_preprocessor(), "cols", FEATURE_COLS)
    print("TODO 5: calibration_curve picture")


if __name__ == "__main__":
    main()
