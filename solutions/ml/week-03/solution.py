"""Week 03 reference solution — SQL / as_of.

Run from the repo root:

    python solutions/ml/week-03/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir
from pipelines.features import build_features

DATA = find_data_dir()
AS_OF = pd.Timestamp("2024-06-01")


def main() -> None:
    usage = pd.read_csv(DATA / "feature_usage.csv", usecols=["date"], parse_dates=["date"])
    events = pd.read_csv(DATA / "user_events.csv", usecols=["timestamp"], parse_dates=["timestamp"])
    subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["signup_date", "churn_date"])

    print("1. Date bound — usage rows on either side of as_of")
    n_cut = int((usage["date"] <= AS_OF).sum())
    n_all = len(usage)
    print(f"  feature_usage date <= {AS_OF.date()}: {n_cut:,}")
    print(f"  feature_usage all rows (what load_customer_360 uses): {n_all:,}")

    print("\n2. Grain test")
    df = build_features(as_of=AS_OF, n=None, at_risk_only=True)
    at_risk = subs[(subs["signup_date"] <= AS_OF) & ~((subs["churn_date"].notna()) & (subs["churn_date"] <= AS_OF))]
    print(f"  build_features rows={len(df):,}  unique={df['user_id'].nunique():,}  at_risk_subs={len(at_risk):,}")
    assert df["user_id"].is_unique
    assert len(df) == len(at_risk)

    print("\n3. tenure_days vs tenure_so_far")
    cols = ["user_id", "tenure_so_far", "tenure_days", "is_churned"]
    print(df[cols].head(5).to_string(index=False))
    disagree = df[df["tenure_so_far"] != df["tenure_days"]]
    print(f"  disagree on {len(disagree):,} / {len(df):,} rows (later churners, or snapshot ≠ as_of)")

    print("\n4. Freshness — latest legal as_of")
    print(f"  usage  min={usage['date'].min()}  max={usage['date'].max()}")
    print(f"  events min={events['timestamp'].min()}  max={events['timestamp'].max()}")
    latest = min(usage["date"].max(), events["timestamp"].max())
    print(f"  latest legal as_of in this universe: {pd.Timestamp(latest).date()} (fixture clips at 2024-11-30)")


if __name__ == "__main__":
    main()
