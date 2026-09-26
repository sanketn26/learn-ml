"""Score tonight's at-risk accounts. Writes a CSV, not an HTTP API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.contract import load_artifact, validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features


def _payload(row: dict) -> dict:
    out = {}
    for key in FEATURE_COLS:
        val = row[key]
        if key == "plan_type":
            out[key] = str(val)
        else:
            out[key] = val.item() if hasattr(val, "item") else val
    return out


def labels_known_by(metrics: dict) -> pd.Timestamp:
    """The last date whose churn this model's evaluation needed."""
    if "labels_known_by" in metrics:
        return pd.Timestamp(metrics["labels_known_by"])
    return pd.Timestamp(metrics["as_of"]) + pd.Timedelta(days=metrics["horizon_days"])


def score_batch(as_of: str, artifact_dir: Path, limit: int = 80) -> pd.DataFrame:
    art = load_artifact(artifact_dir)
    known_by = labels_known_by(art["metrics"])
    if pd.Timestamp(as_of) < known_by:
        # A model backtested as of June 1 was graded on who churned by July 1. On a June 1
        # morning nobody knows that yet: the job that trained it could not have run.
        raise ValueError(
            f"model {art['metrics']['model_version']} was evaluated on churn up to {known_by.date()}; "
            f"it cannot score {pd.Timestamp(as_of).date()}. Train as of score date − horizon (pipelines/job.py)."
        )
    frame = build_features(as_of=as_of, n=None, at_risk_only=True)
    for rec in frame[FEATURE_COLS].to_dict(orient="records"):
        validate(_payload(rec))
    scores = art["pipeline"].predict_proba(frame[FEATURE_COLS])[:, 1]
    cut = art["metrics"]["threshold"]
    ranked = frame[["user_id"]].copy()
    ranked["churn_score"] = scores
    ranked["flag_for_cs"] = ranked["churn_score"] >= cut
    ranked["model_version"] = art["metrics"]["model_version"]
    # Rank on the raw score, ties broken by user_id: the same model and data
    # always ship the same list. Round only for display.
    ranked = ranked.sort_values(["churn_score", "user_id"], ascending=[False, True], kind="stable").head(limit)
    return ranked.assign(churn_score=ranked["churn_score"].round(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(AS_OF_DEFAULT.date()))
    parser.add_argument("--artifact", required=True, help="artifacts/YYYYMMDD or artifacts/prod")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", default="tonight.csv")
    args = parser.parse_args()
    table = score_batch(args.as_of, Path(args.artifact), limit=args.limit)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}  n={len(table)}  version={table['model_version'].iloc[0] if len(table) else '?'}")


if __name__ == "__main__":
    main()
