import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lib.course_data import find_data_dir, load_customer_360  # noqa: E402

DATA = find_data_dir()
df = load_customer_360(DATA)


def task_1():
    adopt = df[df["signup_date"].notna()].groupby("plan_type")["features_adopted"].mean().sort_values()
    fig, ax = plt.subplots()
    ax.bar(adopt.index, adopt.values)
    top = adopt.idxmax()
    ax.annotate(f"{adopt[top]:.2f}", (top, adopt[top]))
    return adopt, fig


def task_2():
    events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"]).dropna()
    region = events.groupby("user_id")["region"].agg(lambda s: s.mode().iat[0]).rename("region")
    by_user = df.merge(region, on="user_id", how="left")
    churn = by_user.groupby(by_user["region"].fillna("unknown"))["is_churned"].mean().sort_values()
    fig, ax = plt.subplots()
    ax.barh(churn.index, churn.values)
    return churn, fig


def task_3():
    plan_churn = df.groupby("plan_type")["is_churned"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.bar(plan_churn.index, plan_churn.values)
    ax.set_ylim(0, plan_churn.max() * 1.15)
    ax.set_title("Free customers churn about twice as often as paid ones")
    return fig
