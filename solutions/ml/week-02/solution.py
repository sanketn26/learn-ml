"""Week 02 reference solution — Pandas as SQL.

Run from the repo root:

    python solutions/ml/week-02/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir

DATA = find_data_dir()


def most_common(series: pd.Series) -> object:
    counts = series.value_counts(dropna=False)
    return counts.index[0] if len(counts) else pd.NA


def assert_join_ok(left: pd.DataFrame, out: pd.DataFrame, key: str, metric: str) -> None:
    in_rows, out_rows = len(left), len(out)
    dupes = int(out.duplicated(key).sum())
    null_share = float(out[metric].isna().mean()) if metric in out.columns else float("nan")
    print(f"  in={in_rows:,}  out={out_rows:,}  dup_{key}={dupes}  null_{metric}={null_share:.3f}")
    if out_rows > 1.01 * in_rows:
        raise ValueError(f"join exploded: {out_rows} rows from {in_rows}")
    if dupes:
        raise ValueError(f"duplicate {key}: {dupes}")


def main() -> None:
    subs = pd.read_csv(DATA / "subscriptions.csv")
    events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"])

    print("1. Plan snapshot")
    snap = (
        subs.groupby("plan_type")
        .agg(users=("user_id", "count"), churn_rate=("is_churned", "mean"), arpu=("mrr", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    print(snap.round(3).to_string())
    leaky = snap["churn_rate"].idxmax()
    print(f"  leaky bucket: {leaky}")

    print("\n2. Region mix (one region per user, then left-join)")
    region = events.groupby("user_id", sort=False)["region"].agg(most_common).rename("region")
    mixed = subs.merge(region, on="user_id", how="left")
    by_region = (
        mixed.groupby(mixed["region"].fillna("unknown"))
        .agg(users=("user_id", "count"), churn_rate=("is_churned", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    print(by_region.round(3).to_string())

    print("\n3. Join validator")
    assert_join_ok(subs, mixed, key="user_id", metric="region")


if __name__ == "__main__":
    main()
