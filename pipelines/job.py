"""The weekly job, with its three dates kept apart.

    python -m pipelines.job --score-date 2024-07-01

score_date   The Monday the list is for. Nothing after it exists — no events, no churn.
train as_of  score_date − horizon. The latest snapshot whose customers we have watched for
             a full horizon by score_date. `train` learns on the snapshot one horizon before
             that, and is graded on this one: every label it touches is known by score_date.
scoring      The score_date snapshot, scored by whatever the gate left in prod.

A job that trains and scores on the same date only works looking back: its
backtest reads churn from the month after the morning it claims to run on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.labels import HORIZON_DAYS
from pipelines.promote import promote
from pipelines.score_batch import score_batch
from pipelines.train import train


def train_as_of(score_date: str | pd.Timestamp, horizon_days: int = HORIZON_DAYS) -> pd.Timestamp:
    """The latest backtest date whose labels have matured by score_date."""
    return pd.Timestamp(score_date) - pd.Timedelta(days=horizon_days)


def weekly(score_date: str, out_dir: Path, prod: Path, horizon_days: int = HORIZON_DAYS, limit: int = 80) -> dict:
    as_of = train_as_of(score_date, horizon_days)
    meta = train(str(as_of.date()), out_dir, horizon_days=horizon_days)
    candidate = Path(out_dir) / meta["model_version"]
    try:
        promote(candidate, prod)
        promoted = True
    except SystemExit as refused:  # the gate said no: keep serving the current prod
        print(refused)
        promoted = False
    if not prod.exists():
        raise SystemExit("no model in prod and the candidate was refused; nothing to score")
    tonight = score_batch(str(pd.Timestamp(score_date).date()), prod, limit=limit)
    return {"score_date": str(pd.Timestamp(score_date).date()), "train_as_of": meta["as_of"],
            "labels_known_by": meta["labels_known_by"], "promoted": promoted, "tonight": tonight}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on matured labels, gate, promote, score the score date.")
    parser.add_argument("--score-date", required=True)
    parser.add_argument("--out", default=str(ROOT / "artifacts"))
    parser.add_argument("--prod", default=str(ROOT / "artifacts" / "prod"))
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--csv", default="tonight.csv")
    args = parser.parse_args()
    result = weekly(args.score_date, Path(args.out), Path(args.prod), args.horizon_days, args.limit)
    result.pop("tonight").to_csv(args.csv, index=False)
    print(result, "→", args.csv)


if __name__ == "__main__":
    main()
