from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipelines.features import NUMERIC, build_features

df = build_features(as_of="2024-06-01")


def task_1():
    X, y = df[NUMERIC], df["is_churned"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    i = NUMERIC.index("mrr")
    return StandardScaler().fit(X).mean_[i], StandardScaler().fit(X_train).mean_[i], len(X), len(X_train)


def task_2():
    return df.assign(has_usage=df["total_usage"] > 0).groupby("has_usage")["is_churned"].agg(["count", "mean"])


def task_3():
    NUM = (int, float)
    REQUIRED = {"mrr": NUM, "tenure_so_far": NUM, "log_usage": NUM, "plan_type": (str,)}

    def assert_score_payload(payload):
        missing, extra = REQUIRED.keys() - payload.keys(), payload.keys() - REQUIRED.keys()
        if missing or extra:
            raise ValueError(f"missing={sorted(missing)} extra={sorted(extra)}")
        for key, types in REQUIRED.items():
            if not isinstance(payload[key], types):
                raise TypeError(f"{key} should be {types}")
    return assert_score_payload
