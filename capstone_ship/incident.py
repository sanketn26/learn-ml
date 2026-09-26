"""Step 7 — the incident night.

Two weeks after the churn score ships, Monday's list looks wrong. The model
did not change. Something upstream did. `incident_frame(as_of, seed)` returns
the feature frame the nightly job actually scored that night: the honest
`build_features` output with one seeded pipeline defect applied.

Every defect is chosen to slip past `pipelines.contract.validate` — the
types are right, the keys are right, nothing is NaN. That is the Week 17
lesson: incidents are joins, units, and stale extracts, not "the model
drifted." You find it with row counts, column histograms, and a fixture
user, then prove it with `diagnose(seed, columns)`.

Reading this file before you've written the postmortem is reading the
answer key. The grader checks the columns you name, not whether you
guessed a defect id.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from pipelines.features import AS_OF_DEFAULT, build_features

STALE_DAYS = 60


def _mrr_in_cents(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    # Billing's export switched from dollars to cents; the schema still says float.
    out = frame.copy()
    out["mrr"] = out["mrr"] * 100
    return out


def _events_fanout(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    # A new device table joined onto events without deduping: every event twice.
    out = frame.copy()
    out["total_events"] = out["total_events"] * 2
    out["n_support"] = out["n_support"] * 2
    return out


def _usage_extract_stale(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    # The usage extract stopped landing STALE_DAYS ago and nobody alerted.
    # Customers who signed up since then have no usage rows at all.
    out = frame.copy()
    recent = out["signup_date"] > as_of - pd.Timedelta(days=STALE_DAYS)
    out.loc[recent, "log_usage"] = 0.0
    out.loc[recent, "total_usage"] = 0.0
    out.loc[recent, "features_adopted"] = 0.0
    return out


DEFECTS = {
    "mrr_in_cents": (_mrr_in_cents, frozenset({"mrr"})),
    "events_fanout": (_events_fanout, frozenset({"total_events", "n_support"})),
    "usage_extract_stale": (_usage_extract_stale, frozenset({"log_usage", "features_adopted"})),
}


def _defect_for(seed: int) -> str:
    digest = hashlib.sha256(f"cloudwave-incident-{seed}".encode()).hexdigest()
    return sorted(DEFECTS)[int(digest, 16) % len(DEFECTS)]


def incident_frame(as_of: str | pd.Timestamp | None = None, seed: int = 0, n: int | None = None) -> pd.DataFrame:
    """The frame the nightly job scored on the incident night."""
    as_of = pd.Timestamp(as_of or AS_OF_DEFAULT)
    clean = build_features(as_of=as_of, n=n)
    apply, _ = DEFECTS[_defect_for(seed)]
    return apply(clean, as_of)


def diagnose(seed: int, columns: set[str] | list[str]) -> bool:
    """True if `columns` is exactly the set of FEATURE_COLS the defect corrupted."""
    _, corrupted = DEFECTS[_defect_for(seed)]
    return frozenset(columns) == corrupted


def reveal(seed: int) -> str:
    """For the reference solution and instructors — not for your first attempt."""
    return _defect_for(seed)


def column_shift(reference: pd.DataFrame, tonight: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean, zero share, and p95 per column on both nights — the first table a postmortem needs."""
    rows = []
    for col in columns:
        ref, now = reference[col].astype(float), tonight[col].astype(float)
        rows.append({
            "column": col,
            "ref_mean": ref.mean(),
            "tonight_mean": now.mean(),
            "mean_ratio": now.mean() / ref.mean() if ref.mean() else np.nan,
            "ref_zero_share": (ref == 0).mean(),
            "tonight_zero_share": (now == 0).mean(),
            "ref_p95": ref.quantile(0.95),
            "tonight_p95": now.quantile(0.95),
        })
    return pd.DataFrame(rows).set_index("column")
