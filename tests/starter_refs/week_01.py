import numpy as np
import pandas as pd

from lib.course_data import find_data_dir

usage = pd.read_csv(find_data_dir() / "feature_usage.csv")


def task_1():
    totals = usage.groupby("feature_name")["usage_count"].sum().to_numpy(dtype=float)
    return {"mean": np.mean(totals), "median": np.median(totals), "p90": np.quantile(totals, 0.9)}


def task_2():
    rng = np.random.default_rng(0)
    some = rng.choice(usage["user_id"].unique(), size=2000, replace=False)
    pivot = usage[usage["user_id"].isin(some)].pivot_table(
        index="user_id", columns="feature_name", values="usage_count", aggfunc="sum", fill_value=0)
    mat = pivot.to_numpy(dtype=float)
    means = mat.mean(axis=1, keepdims=True)
    return pivot, mat / np.where(means == 0, 1, means)


def task_3():
    per_user = usage.groupby("user_id")["usage_count"].sum().astype(float)
    cut = np.quantile(per_user.to_numpy(), 0.99)
    whales = per_user[per_user >= cut]
    return list(whales.index), whales.sum() / per_user.sum()
