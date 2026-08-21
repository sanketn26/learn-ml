"""One-off repair for CloudWave fixtures. Run from the repo root.

Makes tenure_days a real clock, clips billing to the usage/events observation
end (2024-11-30), and aligns feedback category/text/sentiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OBSERVATION_END = pd.Timestamp("2024-11-30")

TEXT_TO_CATEGORY = {
    "Great update — the dashboard is much faster now!": ("praise", 0.85),
    "Not sure how to enable the new feature.": ("other", 0.05),
    "Would love an API to export user timelines.": ("feature_request", 0.35),
    "The app crashes when I upload a file larger than 5MB.": ("bug", -0.75),
    "I was charged twice for my subscription this month.": ("billing", -0.65),
}


def repair_subscriptions() -> set[str]:
    path = DATA / "subscriptions.csv"
    subs = pd.read_csv(path, parse_dates=["signup_date", "churn_date"])
    before = len(subs)
    subs = subs[subs["signup_date"] <= OBSERVATION_END].copy()
    future_churn = subs["churn_date"].notna() & (subs["churn_date"] > OBSERVATION_END)
    subs.loc[future_churn, "churn_date"] = pd.NaT
    subs["is_churned"] = subs["churn_date"].notna().astype(int)
    end = subs["churn_date"].fillna(OBSERVATION_END)
    subs["tenure_days"] = (end - subs["signup_date"]).dt.days.clip(lower=0).astype(int)
    subs.to_csv(path, index=False)
    print(f"subscriptions: {before} -> {len(subs)} rows, churn rate {subs['is_churned'].mean():.4f}")
    return set(subs["user_id"])


def repair_dependent_csv(name: str, valid_users: set[str]) -> None:
    path = DATA / name
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["user_id"].isin(valid_users)]
    df.to_csv(path, index=False)
    print(f"{name}: {before} -> {len(df)} rows")


def repair_feedback(valid_users: set[str]) -> None:
    path = DATA / "feedback.json"
    rows = []
    unknown = 0
    before = 0
    with path.open() as f:
        for line in f:
            before += 1
            row = json.loads(line)
            if row.get("user_id") not in valid_users:
                continue
            text = row.get("feedback_text", "")
            mapped = TEXT_TO_CATEGORY.get(text)
            if mapped is None:
                unknown += 1
                category, sentiment = "other", 0.0
            else:
                category, sentiment = mapped
            row["category"] = category
            row["sentiment_score"] = sentiment
            rows.append(row)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"feedback: {before} -> {len(rows)} lines, unknown templates {unknown}")


def main() -> None:
    valid_users = repair_subscriptions()
    repair_dependent_csv("user_events.csv", valid_users)
    repair_dependent_csv("feature_usage.csv", valid_users)
    repair_feedback(valid_users)
    print("observation_end", OBSERVATION_END.date(), "customers", len(valid_users))


if __name__ == "__main__":
    main()
