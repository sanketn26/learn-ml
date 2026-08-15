"""Shared CloudWave loaders for exercises. No Jupyter, no IPython."""

from __future__ import annotations

from pathlib import Path

LAPTOP_N = 8_000
LAPTOP_SEQ_N = 3_000


def find_data_dir() -> Path:
    here = Path.cwd()
    candidates = [
        here / "data",
        here.parent / "data",
        here.parent.parent / "data",
        Path(__file__).resolve().parent.parent / "data",
    ]
    for candidate in candidates:
        if (candidate / "subscriptions.csv").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find data/subscriptions.csv. Run from the repo root "
        "or an exercises/ folder."
    )


def load_customer_360(data=None, n: int | None = LAPTOP_N, random_state: int = 42):
    """One row per user. Samples for laptop time. Set n=None for all 50k."""
    import numpy as np
    import pandas as pd

    data = Path(data) if data is not None else find_data_dir()
    subs = pd.read_csv(
        data / "subscriptions.csv",
        usecols=["user_id", "plan_type", "mrr", "signup_date", "churn_date", "is_churned", "tenure_days"],
        parse_dates=["signup_date", "churn_date"],
    )
    usage = pd.read_csv(
        data / "feature_usage.csv",
        usecols=["user_id", "feature_name", "usage_count", "avg_session_seconds"],
    )
    events = pd.read_csv(
        data / "user_events.csv",
        usecols=["event_id", "user_id", "event_type", "device", "region"],
    )

    usage_u = usage.groupby("user_id", sort=False).agg(
        total_usage=("usage_count", "sum"),
        features_adopted=("feature_name", "nunique"),
        avg_session=("avg_session_seconds", "mean"),
    )
    events["is_support"] = events["event_type"].eq("support_message")
    events["is_cancel"] = events["event_type"].eq("cancel")
    events_u = events.groupby("user_id", sort=False).agg(
        total_events=("event_id", "count"),
        n_devices=("device", "nunique"),
        n_regions=("region", "nunique"),
        n_support=("is_support", "sum"),
        n_cancels=("is_cancel", "sum"),
    )

    df = subs.merge(usage_u, on="user_id", how="left").merge(events_u, on="user_id", how="left")
    for col in [
        "total_usage",
        "features_adopted",
        "avg_session",
        "total_events",
        "n_devices",
        "n_regions",
        "n_cancels",
        "n_support",
    ]:
        df[col] = df[col].fillna(0)
    df["log_usage"] = np.log1p(df["total_usage"])
    if n is not None and len(df) > n:
        df = df.sample(n, random_state=random_state).reset_index(drop=True)
    return df


def load_weekly_usage_grid(data=None, n_users: int = LAPTOP_SEQ_N, n_weeks: int = 12, random_state: int = 0):
    """Users × last-N-weeks usage matrix + churn labels."""
    import numpy as np
    import pandas as pd

    data = Path(data) if data is not None else find_data_dir()
    usage = pd.read_csv(
        data / "feature_usage.csv",
        usecols=["user_id", "usage_count", "date"],
        parse_dates=["date"],
    )
    labels = pd.read_csv(data / "subscriptions.csv", usecols=["user_id", "is_churned"]).set_index("user_id")[
        "is_churned"
    ]
    usage["week"] = usage["date"].dt.to_period("W").dt.start_time
    weekly = usage.groupby(["user_id", "week"], sort=False)["usage_count"].sum().reset_index()
    weeks = sorted(weekly["week"].unique())[-n_weeks:]
    grid = (
        weekly[weekly["week"].isin(weeks)]
        .pivot_table(index="user_id", columns="week", values="usage_count", fill_value=0)
        .reindex(columns=weeks, fill_value=0)
    )
    common = grid.index.intersection(labels.index)
    grid = grid.loc[common]
    y = labels.loc[common].to_numpy(dtype=np.int64)
    if n_users is not None and len(grid) > n_users:
        rng = np.random.default_rng(random_state)
        take = rng.choice(len(grid), size=n_users, replace=False)
        grid = grid.iloc[take]
        y = y[take]
    X = np.log1p(grid.to_numpy(dtype=np.float32))
    return X, y
