"""One function builds the row training and scoring both use.

Every aggregate is cut at `as_of`. If a column would not exist at noon on
that day, it does not exist here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.course_data import LAPTOP_N, find_data_dir

AS_OF_DEFAULT = pd.Timestamp("2024-06-01")
OBSERVATION_END = pd.Timestamp("2024-11-30")

NUMERIC = [
    "mrr",
    "tenure_so_far",
    "log_usage",
    "features_adopted",
    "total_events",
    "n_support",
]
CATEGORICAL = ["plan_type"]
FEATURE_COLS = NUMERIC + CATEGORICAL

# Never a model input. Keys and labels live beside the frame, not in X.
FORBIDDEN = [
    "user_id",
    "email",
    "churn_date",
    "is_churned",
    "tenure_days",
    "feedback_text",
    "already_churned",
]


def build_features(
    as_of: str | pd.Timestamp | None = None,
    n: int | None = LAPTOP_N,
    data: Path | None = None,
    random_state: int = 42,
    at_risk_only: bool = True,
) -> pd.DataFrame:
    """One row per customer as of `as_of`. Usage and events after that instant are dropped."""
    as_of = pd.Timestamp(as_of or AS_OF_DEFAULT)
    data = Path(data) if data is not None else find_data_dir()

    subs = pd.read_csv(
        data / "subscriptions.csv",
        usecols=["user_id", "plan_type", "mrr", "signup_date", "churn_date", "is_churned", "tenure_days"],
        parse_dates=["signup_date", "churn_date"],
    )
    usage = pd.read_csv(
        data / "feature_usage.csv",
        usecols=["user_id", "feature_name", "usage_count", "date"],
        parse_dates=["date"],
    )
    events = pd.read_csv(
        data / "user_events.csv",
        usecols=["event_id", "user_id", "event_type", "timestamp"],
        parse_dates=["timestamp"],
    )

    signed_up = subs[subs["signup_date"] <= as_of].copy()
    signed_up["already_churned"] = signed_up["churn_date"].notna() & (signed_up["churn_date"] <= as_of)
    if at_risk_only:
        signed_up = signed_up.loc[~signed_up["already_churned"]].copy()

    signed_up["tenure_so_far"] = (as_of - signed_up["signup_date"]).dt.days.clip(lower=0)

    usage = usage[usage["date"] <= as_of]
    events = events[events["timestamp"] <= as_of]

    usage_u = usage.groupby("user_id", sort=False).agg(
        total_usage=("usage_count", "sum"),
        features_adopted=("feature_name", "nunique"),
    )
    events["is_support"] = events["event_type"].eq("support_message")
    events_u = events.groupby("user_id", sort=False).agg(
        total_events=("event_id", "count"),
        n_support=("is_support", "sum"),
    )

    df = signed_up.merge(usage_u, on="user_id", how="left").merge(events_u, on="user_id", how="left")
    for col in ["total_usage", "features_adopted", "total_events", "n_support"]:
        df[col] = df[col].fillna(0)
    df["log_usage"] = np.log1p(df["total_usage"])
    df["as_of"] = as_of

    if n is not None and len(df) > n:
        df = df.sample(n, random_state=random_state).reset_index(drop=True)
    return df


def assert_no_forbidden(frame: pd.DataFrame) -> None:
    leaked = [c for c in FORBIDDEN if c in frame.columns and c in FEATURE_COLS]
    if leaked:
        raise AssertionError(f"forbidden columns in the model matrix: {leaked}")
