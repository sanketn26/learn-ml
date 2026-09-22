"""A brief is the business definition of "shipped."

The model is the same in every brief: one churn score per at-risk customer.
What changes is who is eligible, how many accounts someone can act on, how
the list is ordered, and what counts as success. `select` and `judge` are
the only code path — a new brief is data, not a new pipeline.

The default is Priya's retention desk: 80 calls a week, ranked by risk,
judged by how many of those calls reach someone who was about to leave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


def _everyone(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def _by_risk(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    return scores


def _desk_metrics(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series) -> dict:
    hits = int(y.loc[picked.index].sum())
    return {
        "picked": len(picked),
        "hits": hits,
        "precision": hits / max(len(picked), 1),
        "recall": hits / max(int(y.sum()), 1),
        "base_rate": float(y.mean()),
    }


@dataclass(frozen=True)
class Brief:
    key: str
    stakeholder: str
    question: str
    capacity: int
    success: str
    eligible: Callable[[pd.DataFrame], pd.Series] = field(default=_everyone, repr=False)
    priority: Callable[[pd.DataFrame, np.ndarray], np.ndarray] = field(default=_by_risk, repr=False)
    metrics: Callable[[pd.DataFrame, pd.DataFrame, pd.Series], dict] = field(default=_desk_metrics, repr=False)


RETENTION_DESK = Brief(
    key="retention-desk",
    stakeholder="Priya (CS lead)",
    question="Which 80 at-risk customers should the desk call this week?",
    capacity=80,
    success="precision of the 80 calls (hits / 80), next to the base rate a random 80 would get",
)

BRIEFS: dict[str, Brief] = {RETENTION_DESK.key: RETENTION_DESK}


def select(frame: pd.DataFrame, scores: np.ndarray, brief: Brief = RETENTION_DESK) -> pd.DataFrame:
    """The list this brief ships: eligible rows, ordered by the brief's priority, cut at capacity."""
    scored = frame.assign(churn_score=np.asarray(scores, dtype=float))
    scored["priority"] = np.asarray(brief.priority(scored, scored["churn_score"].to_numpy()), dtype=float)
    pool = scored.loc[brief.eligible(scored)]
    return pool.sort_values("priority", ascending=False).head(brief.capacity)


def threshold_for(picked: pd.DataFrame) -> float:
    """The score cut that reproduces a capacity-sized list — the number that goes in metrics.json."""
    return float(picked["churn_score"].min()) if len(picked) else 1.0


def judge(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series, brief: Brief = RETENTION_DESK) -> dict:
    """Score a shipped list against what actually happened (backtest labels)."""
    return {"brief": brief.key, **brief.metrics(picked, frame, y)}
