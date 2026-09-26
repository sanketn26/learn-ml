import csv
from collections import Counter
from dataclasses import asdict, dataclass

from lib.course_data import find_data_dir


def task_1():
    heads, churned = Counter(), Counter()
    with open(find_data_dir() / "subscriptions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            heads[row["plan_type"]] += 1
            churned[row["plan_type"]] += row["is_churned"] == "1"
    return {p: churned[p] / heads[p] for p in heads}


@dataclass
class CustomerFeatures:
    user_id: str
    plan_type: str
    mrr: float
    tenure_days: int

    @classmethod
    def from_row(cls, row):
        return cls(row["user_id"], row["plan_type"], float(row["mrr"]), int(row["tenure_days"]))

    def to_payload(self):
        return asdict(self)


class MeanBaseline:
    def __init__(self):
        self.mean_ = None

    def fit(self, y):
        self.mean_ = sum(y) / len(y)
        return self

    def predict(self, n):
        if self.mean_ is None:
            raise RuntimeError("call fit() before predict()")
        return [self.mean_] * n


def add_tag_buggy(user_id, tag, tags=[]):  # noqa: B006 — the foot-gun, on purpose
    tags.append(tag)
    return {"user_id": user_id, "tags": tags}


def add_tag(user_id, tag, tags=None):
    tags = [] if tags is None else list(tags)
    tags.append(tag)
    return {"user_id": user_id, "tags": tags}


def task_2():
    return CustomerFeatures


def task_3():
    return MeanBaseline


def task_4():
    return add_tag_buggy, add_tag
