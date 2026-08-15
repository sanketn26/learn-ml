"""Labels as of a day, with a horizon — not the lifetime `is_churned` flag."""

from __future__ import annotations

import pandas as pd

from pipelines.features import OBSERVATION_END

HORIZON_DAYS = 30


def label_churn_in_horizon(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
    observation_end: str | pd.Timestamp = OBSERVATION_END,
) -> pd.Series:
    """1 if they cancel in (as_of, as_of + horizon]. 0 if still around at the horizon.

    Rows we cannot yet observe through the horizon are NaN (censored).
    Rows already churned at as_of are NaN (not at risk).
    """
    as_of = pd.Timestamp(as_of)
    horizon_end = as_of + pd.Timedelta(days=horizon_days)
    observation_end = pd.Timestamp(observation_end)
    churn = frame["churn_date"]

    already_gone = churn.notna() & (churn <= as_of)
    positive = churn.notna() & (churn > as_of) & (churn <= horizon_end)
    if observation_end < horizon_end:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")

    label = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    knowable = ~already_gone
    label.loc[knowable & positive] = 1.0
    label.loc[knowable & ~positive] = 0.0
    return label


def label_eventual_churn(frame: pd.DataFrame) -> pd.Series:
    """1 if they cancel *after* as_of (already-churned rows must already be gone).

    The 30-day horizon is the product question. This fixture only has tens of
    those events. Eventual-after-as_of is the question the file can actually
    supervise. Say so in metrics.json.
    """
    return frame["is_churned"].astype(int)


def drop_unlabelled(frame: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    keep = labels.notna()
    return frame.loc[keep].copy(), labels.loc[keep].astype(int)
