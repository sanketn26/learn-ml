"""Shared CloudWave loaders for exercises. No Jupyter, no IPython."""

from __future__ import annotations

from pathlib import Path

LAPTOP_N = 8_000


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
    """One row per user. Samples for laptop time. Set n=None for all ~49k."""
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
    feedback = pd.read_json(data / "feedback.json", lines=True)
    feedback_u = feedback.groupby("user_id", sort=False).agg(
        n_feedback=("feedback_text", "count"),
        avg_sentiment=("sentiment_score", "mean"),
    )

    df = (
        subs.merge(usage_u, on="user_id", how="left")
        .merge(events_u, on="user_id", how="left")
        .merge(feedback_u, on="user_id", how="left")
    )
    for col in [
        "total_usage",
        "features_adopted",
        "avg_session",
        "total_events",
        "n_devices",
        "n_regions",
        "n_cancels",
        "n_support",
        "n_feedback",
    ]:
        df[col] = df[col].fillna(0)
    df["has_feedback"] = df["n_feedback"].gt(0).astype(int)
    df["log_usage"] = np.log1p(df["total_usage"])
    if n is not None and len(df) > n:
        df = df.sample(n, random_state=random_state).reset_index(drop=True)
    return df


def load_weekly_usage_grid(
    data=None,
    n_users: int | None = None,
    n_weeks: int = 12,
    random_state: int = 0,
    as_of: str = "2024-06-01",
    horizon_days: int = 30,
):
    """Customers at risk on `as_of` × the `n_weeks` weeks before it, plus the horizon label.

    Row i, column -1 is the week ending on `as_of`. y is "cancelled within
    `horizon_days` after `as_of`" — the same question as Week 8, so nothing
    after the morning leaks into X. `n_users` samples rows for a faster demo;
    the default keeps every at-risk customer (the rare class needs them).
    """
    import numpy as np
    import pandas as pd

    from pipelines.labels import drop_unlabelled, label_churn_in_horizon

    data = Path(data) if data is not None else find_data_dir()
    as_of = pd.Timestamp(as_of)
    subs = pd.read_csv(data / "subscriptions.csv", usecols=["user_id", "signup_date", "churn_date"],
                       parse_dates=["signup_date", "churn_date"])
    at_risk = subs[(subs["signup_date"] <= as_of) & ~(subs["churn_date"] <= as_of)]
    at_risk, y = drop_unlabelled(at_risk, label_churn_in_horizon(at_risk, as_of, horizon_days))

    usage = pd.read_csv(data / "feature_usage.csv", usecols=["user_id", "usage_count", "date"],
                        parse_dates=["date"])
    start = as_of - pd.Timedelta(days=7 * n_weeks)
    usage = usage[(usage["date"] > start) & (usage["date"] <= as_of)]
    col = n_weeks - 1 - ((as_of - usage["date"]).dt.days // 7)
    grid = (
        usage.assign(col=col)
        .pivot_table(index="user_id", columns="col", values="usage_count", aggfunc="sum", fill_value=0)
        .reindex(index=at_risk["user_id"], columns=range(n_weeks), fill_value=0)
    )
    X = np.log1p(grid.to_numpy(dtype=np.float32))
    y = y.to_numpy(dtype=np.int64)
    if n_users is not None and len(X) > n_users:
        take = np.random.default_rng(random_state).choice(len(X), size=n_users, replace=False)
        X, y = X[take], y[take]
    return X, y
