"""Week 15 — time split, contract.predict, capacity threshold.

Run from the repo root:

    python exercises/ml/week-15/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.contract import predict, validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_eventual_churn


def main() -> None:
    df = build_features(as_of=AS_OF_DEFAULT, n=None, at_risk_only=True)
    y = label_eventual_churn(df, AS_OF_DEFAULT)
    df, y = drop_unlabelled(df, y)
    print(f"rows={len(df):,}  positives={int(y.sum())}  cols={FEATURE_COLS}")
    demo = {k: df.iloc[0][k] for k in FEATURE_COLS}
    demo["plan_type"] = str(demo["plan_type"])
    try:
        validate({**demo, "email": "ada@cloudwave.test"})
    except Exception as exc:
        print("unknown key →", type(exc).__name__, exc)
    print("TODO 1: time-based split on signup_date vs a shuffled split")
    print("TODO 2: 80 × predict() latency; response has model_version")
    print("TODO 3: threshold that flags at most 80 customers")
    print("TODO 4: overlay train vs later-signup histograms of tenure_so_far")
    print("TODO 5: one-page write-up in this folder as WRITEUP.md")
    print("prep", make_preprocessor())
    print("predict is", predict)


if __name__ == "__main__":
    main()
