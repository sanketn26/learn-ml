"""The time wall: train on an older snapshot, test on today's.

Do not split one snapshot on `signup_date`. At a fixed `as_of`,
`tenure_so_far = as_of - signup_date`, so a signup cut is a tenure cut:
train gets only old customers, test gets only new ones, and the two never
overlap on the strongest feature in the table. The model learns from a
tenure range it will never score.

A backtest asks the production question instead. Stand at
`as_of - horizon_days`, learn from what happened by `as_of`, then score the
`as_of` snapshot and check what happened in the next `horizon_days`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.features import AS_OF_DEFAULT, OBSERVATION_END, build_features
from pipelines.labels import drop_unlabelled, label_churn_in_horizon

BACKTEST_HORIZON_DAYS = 90


def snapshot_split(
    as_of: str | pd.Timestamp | None = None,
    horizon_days: int = BACKTEST_HORIZON_DAYS,
    n: int | None = None,
    data: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (train, y_train, test, y_test).

    train: at-risk customers at `as_of - horizon_days`, labelled with churn
           in the following `horizon_days` — fully observed by `as_of`.
    test:  at-risk customers at `as_of`, labelled with churn in the next
           `horizon_days` — the future this backtest checks against.
    """
    as_of = pd.Timestamp(as_of or AS_OF_DEFAULT)
    if as_of + pd.Timedelta(days=horizon_days) > OBSERVATION_END:
        raise ValueError(
            f"as_of {as_of.date()} + {horizon_days}d runs past the last log "
            f"({OBSERVATION_END.date()}); every test label would be censored."
        )
    train_as_of = as_of - pd.Timedelta(days=horizon_days)

    train_raw = build_features(as_of=train_as_of, n=n, data=data)
    y_train = label_churn_in_horizon(train_raw, train_as_of, horizon_days, observation_end=as_of)
    train, y_train = drop_unlabelled(train_raw, y_train)

    test_raw = build_features(as_of=as_of, n=n, data=data)
    test, y_test = drop_unlabelled(test_raw, label_churn_in_horizon(test_raw, as_of, horizon_days))
    return train, y_train, test, y_test
