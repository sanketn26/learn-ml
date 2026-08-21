"""Labels as of a day, with a horizon — not the lifetime `is_churned` flag."""

from __future__ import annotations

import pandas as pd

from pipelines.features import AS_OF_DEFAULT, OBSERVATION_END

HORIZON_DAYS = 30


def label_churn_in_horizon(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
    observation_end: str | pd.Timestamp = OBSERVATION_END,
) -> pd.Series:
    """1 if they cancel in (as_of, as_of + horizon]. 0 if still around at the horizon.

    Rows already churned at as_of are NaN (not at risk).
    Rows we have not watched through the horizon are NaN (censored), except
    cancels we already observed inside the window — those are 1.
    """
    as_of = pd.Timestamp(as_of)
    horizon_end = as_of + pd.Timedelta(days=horizon_days)
    observation_end = pd.Timestamp(observation_end)
    churn = pd.to_datetime(frame["churn_date"], errors="coerce")

    already_gone = churn.notna() & (churn <= as_of)
    in_window = churn.notna() & (churn > as_of) & (churn <= horizon_end)
    observed_positive = in_window & (churn <= observation_end)
    watched_through_horizon = observation_end >= horizon_end
    knowable_zero = ~already_gone & ~in_window & watched_through_horizon

    label = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    label.loc[observed_positive] = 1.0
    label.loc[knowable_zero] = 0.0
    return label


def label_eventual_churn(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
) -> pd.Series:
    """1 if they cancel after as_of. Already-churned rows are NaN.

    The 30-day horizon is the product question. This fixture only has tens of
    those events. Eventual-after-as_of is the question the file can actually
    supervise. Say so in metrics.json.
    """
    as_of = pd.Timestamp(as_of or AS_OF_DEFAULT)
    churn = pd.to_datetime(frame["churn_date"], errors="coerce")
    already_gone = churn.notna() & (churn <= as_of)
    later = churn.notna() & (churn > as_of)
    label = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    label.loc[~already_gone & later] = 1.0
    label.loc[~already_gone & ~later] = 0.0
    return label


def drop_unlabelled(frame: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    keep = labels.notna()
    return frame.loc[keep].copy(), labels.loc[keep].astype(int)
