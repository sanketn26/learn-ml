from __future__ import annotations

import pandas as pd
import pytest

from pipelines.features import OBSERVATION_END
from pipelines.split import snapshot_split

AS_OF = pd.Timestamp("2024-06-01")


@pytest.fixture(scope="module")
def split():
    return snapshot_split(AS_OF, horizon_days=90, n=None)


def test_train_snapshot_is_one_horizon_earlier(split):
    train, _, test, _ = split
    assert (train["as_of"] == AS_OF - pd.Timedelta(days=90)).all()
    assert (test["as_of"] == AS_OF).all()


def test_train_labels_are_knowable_at_as_of(split):
    train, y_train, _, _ = split
    churn = pd.to_datetime(train.loc[y_train == 1, "churn_date"])
    assert (churn <= AS_OF).all()


def test_tenure_ranges_overlap(split):
    """A signup_date cut gives disjoint tenure ranges; a backtest must not."""
    train, _, test, _ = split
    assert train["tenure_so_far"].min() <= test["tenure_so_far"].max()
    assert test["tenure_so_far"].min() <= train["tenure_so_far"].max()


def test_both_sides_have_positives(split):
    _, y_train, _, y_test = split
    assert y_train.sum() > 0 and y_test.sum() > 0


def test_refuses_a_fully_censored_test_window():
    with pytest.raises(ValueError, match="censored"):
        snapshot_split(OBSERVATION_END - pd.Timedelta(days=10), horizon_days=90)
