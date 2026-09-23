"""A brief is the business definition of "shipped."

The model is the same in every brief: one churn score per at-risk customer.
What changes is who is eligible, how many accounts someone can act on, how
the list is ordered, and what counts as success. `select` and `judge` are
the only code path — a new brief is data, not a new pipeline.

The default is Priya's retention desk: 80 calls a week, ranked by risk,
judged by how many of those calls reach someone who was about to leave.
The other briefs are the scenario bank (docs/ml/capstone-scenarios.md):
the same scores, four other business definitions of success.
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
    # Optional money cap: rows are taken in priority order while cumulative cost fits the budget.
    cost: Callable[[pd.DataFrame], pd.Series] | None = field(default=None, repr=False)
    budget: float | None = None


RETENTION_DESK = Brief(
    key="retention-desk",
    stakeholder="Priya (CS lead)",
    question="Which 80 at-risk customers should the desk call this week?",
    capacity=80,
    success="precision of the 80 calls (hits / 80), next to the base rate a random 80 would get",
)

# --- scenario bank -----------------------------------------------------------

DISCOUNT_MONTHS, DISCOUNT_RATE = 3, 0.20
VALUE_MONTHS = 12  # stated assumption: a saved customer is worth a year of MRR


def _discount_cost(frame: pd.DataFrame) -> pd.Series:
    return frame["mrr"] * DISCOUNT_RATE * DISCOUNT_MONTHS


def _discount_metrics(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series) -> dict:
    spend = float(_discount_cost(picked).sum())
    churner_mrr = float(picked.loc[y.loc[picked.index] == 1, "mrr"].sum())
    return {
        **_desk_metrics(picked, frame, y),
        "discount_spend": round(spend, 2),
        "churner_mrr_reached": round(churner_mrr, 2),
        # The offer pays for itself if at least this share of the churners it reached stay because
        # of it, valuing a saved customer at VALUE_MONTHS of MRR. Above 1.0 it cannot pay for itself.
        "break_even_save_rate": round(spend / (churner_mrr * VALUE_MONTHS), 3) if churner_mrr else None,
    }


DISCOUNT_TARGETING = Brief(
    key="discount-targeting",
    stakeholder="Helen (CFO)",
    question="Where should a $3,000 retention-discount budget go: 20% off for three months?",
    capacity=500,
    success="break-even save rate — the share of reached churners who must stay for the discount to pay for itself",
    eligible=lambda f: f["mrr"] > 0,
    # A proportional discount costs k·mrr and protects mrr, so return per dollar is
    # score / k whatever the account's size: rank by risk, not by risk × MRR.
    metrics=_discount_metrics,
    cost=_discount_cost,
    budget=3000.0,
)

UPGRADE_PLANS = {"free", "starter", "pro"}


def _expansion_metrics(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series) -> dict:
    # Guardrail, not hit rate: how many of the pitched accounts were about to leave,
    # versus a usage-only list of the same size.
    pool = frame.loc[frame["plan_type"].isin(UPGRADE_PLANS)]
    usage_only = pool.sort_values("log_usage", ascending=False).head(len(picked))
    return {
        "picked": len(picked),
        "churners_pitched": int(y.loc[picked.index].sum()),
        "churners_pitched_usage_only": int(y.loc[usage_only.index].sum()),
        "median_log_usage": round(float(picked["log_usage"].median()), 3),
    }


EXPANSION_RANKING = Brief(
    key="expansion-ranking",
    stakeholder="Marcus (PM)",
    question="Sales can pitch an upgrade to 50 engaged accounts. Which 50 — without pitching anyone about to leave?",
    capacity=50,
    success="churners pitched (lower is better) against a usage-only list of the same size",
    eligible=lambda f: f["plan_type"].isin(UPGRADE_PLANS),
    priority=lambda f, s: f["log_usage"].to_numpy() * (1 - s),
    metrics=_expansion_metrics,
)


def _slice_metrics(eligible: Callable[[pd.DataFrame], pd.Series]):
    def metrics(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series) -> dict:
        in_slice = eligible(frame)
        y_slice = y.loc[in_slice[in_slice].index]
        hits = int(y.loc[picked.index].sum())
        return {
            "picked": len(picked),
            "slice_size": int(in_slice.sum()),
            "hits": hits,
            "precision": hits / max(len(picked), 1),
            "recall_in_slice": hits / max(int(y_slice.sum()), 1),
            "slice_base_rate": float(y_slice.mean()) if len(y_slice) else 0.0,
        }
    return metrics


def _has_tickets(frame: pd.DataFrame) -> pd.Series:
    return frame["n_support"] > 0


SUPPORT_DEFLECTION = Brief(
    key="support-deflection",
    stakeholder="Ana (support on-call)",
    question="30 senior-agent slots a week. Whose open tickets go to the senior queue?",
    capacity=30,
    success="recall of churners among customers with tickets, next to the ticket-holders' base rate",
    eligible=_has_tickets,
    metrics=_slice_metrics(_has_tickets),
)

NEW_ACCOUNT_DAYS = 45


def _new_accounts(frame: pd.DataFrame) -> pd.Series:
    return frame["tenure_so_far"] <= NEW_ACCOUNT_DAYS


ONBOARDING_ACTIVATION = Brief(
    key="onboarding-activation",
    stakeholder="Priya (CS lead)",
    question=f"40 guided onboarding sessions a week for accounts in their first {NEW_ACCOUNT_DAYS} days. Who gets one?",
    capacity=40,
    success="precision inside the new-account slice, against the slice's own base rate",
    eligible=_new_accounts,
    metrics=_slice_metrics(_new_accounts),
)

BRIEFS: dict[str, Brief] = {
    b.key: b
    for b in (RETENTION_DESK, DISCOUNT_TARGETING, EXPANSION_RANKING, SUPPORT_DEFLECTION, ONBOARDING_ACTIVATION)
}


def select(frame: pd.DataFrame, scores: np.ndarray, brief: Brief = RETENTION_DESK) -> pd.DataFrame:
    """The list this brief ships: eligible rows, ordered by the brief's priority, cut at capacity."""
    scored = frame.assign(churn_score=np.asarray(scores, dtype=float))
    scored["priority"] = np.asarray(brief.priority(scored, scored["churn_score"].to_numpy()), dtype=float)
    pool = scored.loc[brief.eligible(scored)].sort_values("priority", ascending=False)
    if brief.cost is not None and brief.budget is not None:
        pool = pool.loc[brief.cost(pool).cumsum() <= brief.budget]
    return pool.head(brief.capacity)


def threshold_for(picked: pd.DataFrame) -> float:
    """The score cut that reproduces a capacity-sized list — the number that goes in metrics.json."""
    return float(picked["churn_score"].min()) if len(picked) else 1.0


def judge(picked: pd.DataFrame, frame: pd.DataFrame, y: pd.Series, brief: Brief = RETENTION_DESK) -> dict:
    """Score a shipped list against what actually happened (backtest labels)."""
    return {"brief": brief.key, **brief.metrics(picked, frame, y)}
