"""Week 11 — precision@k vs a SQL sort.

    python exercises/ml/week-11/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pipelines.features import (
    AS_OF_DEFAULT,
    FEATURE_COLS,
    build_features,
    make_preprocessor,
)
from pipelines.labels import drop_unlabelled, label_eventual_churn


def precision_at_k(y, scores, k=80) -> float:
    order = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(y)[order].mean())


def main() -> None:
    # Eventual-after-as_of: this fixture only has tens of 30-day cancels.
    df = build_features(as_of=AS_OF_DEFAULT, n=None)
    y = label_eventual_churn(df, AS_OF_DEFAULT)
    df, y = drop_unlabelled(df, y)
    cut = df["signup_date"].quantile(0.80)
    train, test = df[df["signup_date"] <= cut], df[df["signup_date"] > cut]
    model = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
        ]
    )
    model.fit(train[FEATURE_COLS], y.loc[train.index])
    scores = model.predict_proba(test[FEATURE_COLS])[:, 1]
    y_test = y.loc[test.index]
    print("positives in test", int(y_test.sum()), "of", len(y_test))
    print("model@80   ", precision_at_k(y_test, scores))
    print("support@80 ", precision_at_k(y_test, test["n_support"]))
    print("TODO: k in {20,80,200} and the PM causal trap")


if __name__ == "__main__":
    main()
