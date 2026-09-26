"""Run a validated command against the repo. Every command is read-only.

A `Night` is the world one investigation runs in: the incident night, and
which seeded defects (from capstone_ship/incident.py) corrupted its frame.
Any other date gets the honest `build_features` frame — corrupted data
only exists on the night it broke, the same as in production.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.spec import FEATURE_COLUMNS
from capstone_ship.incident import DEFECTS
from pipelines.contract import load_artifact
from pipelines.features import FEATURE_COLS, FORBIDDEN, build_features
from pipelines.labels import HORIZON_DAYS
from pipelines.ranking import top_k

ROOT = Path(__file__).resolve().parent.parent
PROD_AS_OF = "2024-06-01"
PROD_DIR = ROOT / "artifacts" / "capstone-commands"

DEFECT_CAUSE = {"mrr_in_cents": "unit_change", "events_fanout": "join_fanout", "usage_extract_stale": "stale_extract"}
COHORTS = [("0-60d", 0, 60), ("61-180d", 61, 180), ("181d+", 181, 10**6)]

# What check_leakage knows about a column: does it exist at noon on as_of, and why it is or isn't an input.
# Forbidden-ness itself comes from pipelines.features.FORBIDDEN, the list the pipeline enforces.
LEAKAGE_FACTS = {
    "user_id": (True, "a key, not a signal — it identifies the row"),
    "email": (True, "a key and PII"),
    "churn_date": (False, "the label: it is filled in after the customer leaves"),
    "is_churned": (False, "the label itself"),
    "tenure_days": (False, "lifetime tenure, computed from churn_date — it knows when the customer left"),
    "feedback_text": (False, "written after the fact; dated rows can post-date as_of"),
    "already_churned": (True, "a population filter, not a feature"),
    "total_usage": (True, "cut at as_of; redundant with log_usage"),
    "signup_date": (True, "known at signup; use tenure_so_far instead of a raw date"),
}


class CommandError(RuntimeError):
    """A valid call that cannot run on this night's data (for example, an unknown customer)."""


@lru_cache(maxsize=32)
def clean_frame(as_of: str) -> pd.DataFrame:
    return build_features(as_of=as_of, n=None)


@lru_cache(maxsize=1)
def prod_pipeline():
    """The model in production: trained once as of PROD_AS_OF, cached under artifacts/."""
    version_dir = PROD_DIR / pd.Timestamp(PROD_AS_OF).strftime("%Y%m%d")
    if not (version_dir / "model.joblib").exists():
        from pipelines.train import train

        train(PROD_AS_OF, PROD_DIR)
    return load_artifact(version_dir)["pipeline"]


class Night:
    def __init__(self, as_of: str, defects: tuple[str, ...] = ()):
        self.as_of, self.defects = as_of, tuple(defects)
        self._frames: dict[str, pd.DataFrame] = {}

    def frame(self, as_of: str) -> pd.DataFrame:
        if as_of not in self._frames:
            frame = clean_frame(as_of)
            if as_of == self.as_of:
                for name in self.defects:
                    frame = DEFECTS[name][0](frame, pd.Timestamp(as_of))
            self._frames[as_of] = frame
        return self._frames[as_of]


def _ratio(now: float, ref: float) -> float | None:
    return round(float(now / ref), 3) if ref else None


def check_grain(night: Night, as_of: str) -> dict:
    frame = night.frame(as_of)
    dupes = int(frame["user_id"].duplicated().sum())
    late = int((frame["signup_date"] > pd.Timestamp(as_of)).sum())
    return {"as_of": as_of, "rows": len(frame), "duplicate_user_ids": dupes,
            "signups_after_as_of": late, "grain_ok": dupes == 0 and late == 0}


def _shift_score(row: dict) -> float:
    ratio = row["mean_ratio"] or 1.0
    return max(abs(np.log(max(ratio, 1e-9))), 10 * abs(row["zero_share"] - row["ref_zero_share"]))


def compare_nights(night: Night, columns: list[str], ref_as_of: str, as_of: str) -> dict:
    ref, now = night.frame(ref_as_of), night.frame(as_of)
    shifts = []
    for col in columns:
        r, n = ref[col].astype(float), now[col].astype(float)
        shifts.append({
            "column": col,
            "mean_ratio": _ratio(n.mean(), r.mean()),
            "ref_zero_share": round(float((r == 0).mean()), 3),
            "zero_share": round(float((n == 0).mean()), 3),
            "p95_ratio": _ratio(n.quantile(0.95), r.quantile(0.95)),
        })
    # A fixture for inspect_customer: a *typical* customer in the most-shifted column — one whose own
    # ratio matches the column's — not the outlier who also had a busy week.
    top = max(shifts, key=_shift_score)
    worst, target = top["column"], top["mean_ratio"] or 1.0
    active = ref.loc[(ref[FEATURE_COLUMNS] > 0).all(axis=1), "user_id"]  # every column readable as a ratio
    both = ref[["user_id", worst]].merge(now[["user_id", worst]], on="user_id", suffixes=("_ref", "_now"))
    both = both.loc[(both[f"{worst}_ref"] > 0) & both["user_id"].isin(active)]
    own = both[f"{worst}_now"].astype(float) / both[f"{worst}_ref"].astype(float)
    distance = np.abs(np.log(own.clip(lower=1e-9)) - np.log(target))
    example = both.assign(distance=distance).sort_values(["distance", "user_id"])
    return {"shifts": shifts, "example_user_id": example["user_id"].iloc[0] if len(example) else None}


def _cohort(frame: pd.DataFrame) -> pd.Series:
    labels = pd.Series("181d+", index=frame.index)
    for name, lo, hi in COHORTS:
        labels[frame["tenure_so_far"].between(lo, hi)] = name
    return labels


def slice_column(night: Night, column: str, by: str, ref_as_of: str, as_of: str) -> dict:
    ref, now = night.frame(ref_as_of), night.frame(as_of)
    key = _cohort if by == "signup_cohort" else (lambda f: f["plan_type"])
    ref_keys, now_keys = key(ref), key(now)
    slices = []
    for name in sorted(set(now_keys) | set(ref_keys), key=str):
        r = ref.loc[ref_keys == name, column].astype(float)
        n = now.loc[now_keys == name, column].astype(float)
        slices.append({
            "slice": name, "rows": len(n),
            "ref_zero_share": round(float((r == 0).mean()), 3) if len(r) else None,
            "zero_share": round(float((n == 0).mean()), 3) if len(n) else None,
            "mean_ratio": _ratio(n.mean(), r.mean()) if len(r) and len(n) else None,
        })
    return {"column": column, "by": by, "slices": slices}


def inspect_customer(night: Night, user_id: str, ref_as_of: str, as_of: str) -> dict:
    ref, now = night.frame(ref_as_of), night.frame(as_of)
    r, n = ref.loc[ref["user_id"] == user_id], now.loc[now["user_id"] == user_id]
    if r.empty or n.empty:
        raise CommandError(f"{user_id} is not at risk on both {ref_as_of} and {as_of}")
    rows = []
    for col in FEATURE_COLUMNS:
        a, b = float(r[col].iloc[0]), float(n[col].iloc[0])
        rows.append({"column": col, "ref": round(a, 3), "tonight": round(b, 3), "ratio": _ratio(b, a)})
    return {"user_id": user_id, "plan_type": str(n["plan_type"].iloc[0]), "rows": rows}


def check_leakage(night: Night, column: str) -> dict:
    available, reason = LEAKAGE_FACTS.get(column, (True, "a model input, cut at as_of"))
    return {"column": column, "in_feature_cols": column in FEATURE_COLS, "forbidden": column in FORBIDDEN,
            "available_at_as_of": available, "reason": reason}


RULES = [
    (re.compile(r"missing keys: \[(.*)\]"), "missing_key", "send every key in REQUIRED; do not default a missing feature"),
    (re.compile(r"unknown keys: \[(.*)\]"), "unknown_key", "strip keys, ids and labels before calling predict()"),
    (re.compile(r"(\w+) should be .*, got"), "wrong_type", "cast at the edge, before the payload is built"),
    (re.compile(r"(\w+) is missing"), "nan_value", "fix the upstream fill, never fill inside the contract"),
    (re.compile(r"unknown plan_type (\S+)"), "unknown_plan", "map the plan to a known value upstream"),
]


def explain_rejection(night: Night, error: str) -> dict:
    for pattern, rule, fix in RULES:
        match = pattern.search(error)
        if match:
            if rule == "unknown_plan":
                fields = ["plan_type"]
            else:
                fields = [f.strip(" '\"") for f in match.group(1).split(",") if f.strip()]
            return {"rule": rule, "fields": fields, "fix": fix}
    raise CommandError("not a pipelines.contract.validate() message")


def check_threshold(night: Night, capacity: int, as_of: str) -> dict:
    known_by = pd.Timestamp(PROD_AS_OF) + pd.Timedelta(days=HORIZON_DAYS)
    if pd.Timestamp(as_of) < known_by:
        raise CommandError(f"the production model's labels are known by {known_by.date()}; it did not exist on {as_of}")
    frame = night.frame(as_of)
    scores = prod_pipeline().predict_proba(frame[FEATURE_COLS])[:, 1]
    cut = float(scores[top_k(scores, frame["user_id"], capacity)[-1]])
    return {"capacity": capacity, "threshold": round(cut, 4), "strictly_above": int((scores > cut).sum()),
            "ties_at_threshold": int((scores == cut).sum()), "flagged_at_threshold": int((scores >= cut).sum())}


def conclude(night: Night, **_) -> dict:
    return {"closed": True}


def escalate(night: Night, reason: str) -> dict:
    return {"escalated": reason}


COMMANDS = {f.__name__: f for f in (check_grain, compare_nights, slice_column, inspect_customer, check_leakage,
                                    explain_rejection, check_threshold, conclude, escalate)}


def execute(call: dict, night: Night) -> dict:
    """Run a call that already passed validate_call."""
    return COMMANDS[call["command"]](night, **call["args"])
