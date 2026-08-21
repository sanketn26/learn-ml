"""Week 14 exercises — Week 14 — Neural Nets, Without the Mystique.

Run from the repo root:

    python exercises/ml/week-14/starter.py
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
    print("rows", len(df), "features", FEATURE_COLS)
    print("prep width", make_preprocessor().fit_transform(df[FEATURE_COLS][:50]).shape[1])
    print("TODO 1: MLPClassifier(activation='identity') vs LogisticRegression")
    print("TODO 2: hidden_layer_sizes=(128, 128, 128) train vs test")
    print("TODO 4: comment out opt.zero_grad() for 5 epochs")


if __name__ == "__main__":
    main()
