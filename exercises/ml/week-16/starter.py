"""Week 16 — SQL / as_of extract.

    python exercises/ml/week-16/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import build_features


def main() -> None:
    as_of = "2024-06-01"
    df = build_features(as_of=as_of, n=None)
    print(f"as_of={as_of}  rows={len(df):,}  unique={df['user_id'].nunique():,}")
    print("TODO 1: count feature_usage rows with date <= as_of vs all")
    print("TODO 2: assert unique user_id and match at-risk subscriptions")
    print("TODO 3: compare tenure_days vs tenure_so_far on 5 users")
    print("TODO 4: min/max of usage and events — latest legal as_of")


if __name__ == "__main__":
    main()
