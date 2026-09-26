import duckdb
import pandas as pd

from lib.course_data import find_data_dir
from pipelines.features import build_features

DATA = find_data_dir()
AS_OF = pd.Timestamp("2024-06-01")


def task_1():
    return duckdb.sql(f"""
        SELECT count(*) FILTER (WHERE date <= DATE '2024-06-01'), count(*)
        FROM read_csv_auto('{DATA / "feature_usage.csv"}')
    """).fetchone()


def task_2():
    return build_features(as_of=AS_OF, n=None)


def task_3():
    frame = build_features(as_of=AS_OF, n=None)
    return pd.concat([frame[frame["churn_date"].notna()].head(2), frame[frame["churn_date"].isna()].head(3)])


def task_4():
    usage = pd.read_csv(DATA / "feature_usage.csv", usecols=["date"], parse_dates=["date"])["date"]
    events = pd.read_csv(DATA / "user_events.csv", usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
    return min(usage.max(), events.max().normalize())
