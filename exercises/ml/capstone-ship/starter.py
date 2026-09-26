"""Job-path capstone — ship the churn score.

Run from the repo root:

    python exercises/ml/capstone-ship/starter.py

Fill in the steps in order. The runner stops at the first TODO and tells you
which step is next. Everything writes under artifacts/capstone-ship/.
Reuse pipelines/ — every step has a function there already.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone_ship.briefs import RETENTION_DESK, Brief, judge, select, threshold_for  # noqa: F401
from capstone_ship.incident import column_shift, diagnose, incident_frame  # noqa: F401
from pipelines.contract import load_artifact, predict, validate  # noqa: F401
from pipelines.features import FEATURE_COLS, NUMERIC, build_features  # noqa: F401
from pipelines.promote import gate, promote  # noqa: F401
from pipelines.score_batch import score_batch  # noqa: F401
from pipelines.split import snapshot_split  # noqa: F401
from pipelines.train import train  # noqa: F401

WORKDIR = ROOT / "artifacts" / "capstone-ship"
AS_OF = pd.Timestamp("2024-06-01")  # the backtest date
HORIZON = 30  # TODO (step 2): justify this number in your write-up
SCORE_DATE = AS_OF + pd.Timedelta(days=HORIZON)  # the first morning the backtest's labels exist
INCIDENT_NIGHT = SCORE_DATE + pd.Timedelta(days=14)
SEED = 0  # your incident; any integer


def step1_features(as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """As-of feature frame for every at-risk customer. Assert its grain."""
    raise NotImplementedError("step 1: build_features + grain asserts")


def step2_labels(as_of: pd.Timestamp = AS_OF, horizon: int = HORIZON):
    """(train, y_train, test, y_test) with an explicit horizon. Both sides need positives."""
    raise NotImplementedError("step 2: snapshot_split with your horizon")


def step3_train(out_dir: Path, as_of: pd.Timestamp = AS_OF, horizon: int = HORIZON, n: int | None = None) -> Path:
    """Train a candidate and prove it beats the dummy. Return the candidate directory."""
    raise NotImplementedError("step 3: pipelines.train.train + the promote gate")


def step4_threshold(candidate: Path, test_df: pd.DataFrame, y_test: pd.Series,
                    brief: Brief = RETENTION_DESK) -> dict:
    """Choose the threshold from the brief's capacity, not 0.5. Write it into metrics.json."""
    raise NotImplementedError("step 4: select → judge → threshold_for")


def step5_contract(candidate: Path, test_df: pd.DataFrame) -> dict:
    """Call predict() on a real payload; prove validate() rejects three bad ones."""
    raise NotImplementedError("step 5: predict + three validate() rejections")


def step6_promote_and_score(candidate: Path, prod: Path, score_date: pd.Timestamp = SCORE_DATE,
                            brief: Brief = RETENTION_DESK) -> pd.DataFrame:
    """Promote through the gate, then score the SCORE_DATE list at the brief's capacity."""
    raise NotImplementedError("step 6: promote + score_batch")


def step7_incident(prod: Path, seed: int = SEED, night: pd.Timestamp = INCIDENT_NIGHT) -> dict:
    """Find what broke on the incident night. Return {'suspects': [...], 'diagnosed': bool, ...}."""
    raise NotImplementedError("step 7: incident_frame vs last week → column_shift → diagnose")


def main() -> None:
    out, prod = WORKDIR / "artifacts", WORKDIR / "artifacts" / "prod"
    try:
        frame = step1_features()
        print(f"1. features: {len(frame):,} at-risk customers as of {AS_OF.date()}")
        train_df, y_train, test_df, y_test = step2_labels()
        print(f"2. labels: train positives={int(y_train.sum())}  test positives={int(y_test.sum())}")
        candidate = step3_train(out)
        print(f"3. candidate: {candidate}")
        verdict = step4_threshold(candidate, test_df, y_test)
        print(f"4. brief: {verdict}")
        print(f"5. contract: {step5_contract(candidate, test_df)}")
        tonight = step6_promote_and_score(candidate, prod)
        print(f"6. tonight: {len(tonight)} names")
        incident = step7_incident(prod)
        print(f"7. incident: suspects={incident['suspects']} diagnosed={incident['diagnosed']}")
    except NotImplementedError as todo:
        print(f"\nnext: {todo}")


if __name__ == "__main__":
    main()
