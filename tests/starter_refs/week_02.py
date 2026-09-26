import pandas as pd

from lib.course_data import find_data_dir

DATA = find_data_dir()
subs = pd.read_csv(DATA / "subscriptions.csv")


def task_1():
    return subs.groupby("plan_type").agg(users=("user_id", "count"), churn_rate=("is_churned", "mean"), arpu=("mrr", "mean"))


def task_2():
    events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"])
    region = events.dropna(subset=["region"]).groupby("user_id")["region"].agg(lambda s: s.mode().iat[0]).rename("region")
    return subs.merge(region, on="user_id", how="left")


def task_3():
    def validate_join(left, out, key, metric):
        if len(out) > 1.01 * len(left):
            raise ValueError(f"join fanned out: {len(left)} → {len(out)}")
    return validate_join
