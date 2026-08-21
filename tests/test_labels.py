from __future__ import annotations

import pandas as pd

from pipelines.features import AS_OF_DEFAULT, build_features
from pipelines.labels import drop_unlabelled, label_churn_in_horizon


def test_lifetime_flag_is_not_the_training_label():
    as_of = AS_OF_DEFAULT
    df = build_features(as_of=as_of, n=None, at_risk_only=True)
    y = label_churn_in_horizon(df, as_of)
    labelled, y2 = drop_unlabelled(df, y)
    # Lifetime is_churned includes people who cancel long after the horizon.
    assert (labelled["is_churned"] >= y2).all()
    assert float(y2.mean()) < float(labelled["is_churned"].mean())


def test_already_churned_are_unlabelled():
    as_of = AS_OF_DEFAULT
    gone = pd.DataFrame(
        {
            "churn_date": [as_of - pd.Timedelta(days=3)],
        }
    )
    y = label_churn_in_horizon(gone, as_of)
    assert y.isna().all()


def test_short_observation_window_keeps_seen_cancels():
    as_of = pd.Timestamp("2024-06-01")
    frame = pd.DataFrame(
        {
            "churn_date": [
                as_of + pd.Timedelta(days=5),
                pd.NaT,
            ]
        }
    )
    y = label_churn_in_horizon(
        frame, as_of, horizon_days=30, observation_end=as_of + pd.Timedelta(days=10)
    )
    assert float(y.iloc[0]) == 1.0
    assert pd.isna(y.iloc[1])


def test_eventual_churn_ignores_people_already_gone():
    from pipelines.labels import label_eventual_churn

    as_of = pd.Timestamp("2024-06-01")
    frame = pd.DataFrame(
        {
            "churn_date": [
                as_of - pd.Timedelta(days=3),
                as_of + pd.Timedelta(days=40),
                pd.NaT,
            ]
        }
    )
    y = label_eventual_churn(frame, as_of)
    assert pd.isna(y.iloc[0])
    assert float(y.iloc[1]) == 1.0
    assert float(y.iloc[2]) == 0.0
