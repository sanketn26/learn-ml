"""Score tonight's at-risk accounts. Writes a CSV, not an HTTP API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.contract import load_artifact, predict
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features


def score_batch(as_of: str, artifact_dir: Path, limit: int = 80) -> pd.DataFrame:
    art = load_artifact(artifact_dir)
    frame = build_features(as_of=as_of, n=None, at_risk_only=True)
    scores = art["pipeline"].predict_proba(frame[FEATURE_COLS])[:, 1]
    cut = art["metrics"]["threshold"]
    ranked = frame[["user_id"]].copy()
    ranked["churn_score"] = scores.round(4)
    ranked["flag_for_cs"] = ranked["churn_score"] >= cut
    ranked["model_version"] = art["metrics"]["model_version"]
    # One-row contract check so serve-path validate() still runs in CI.
    predict({k: frame.iloc[0][k] for k in FEATURE_COLS}, art)
    return ranked.sort_values("churn_score", ascending=False).head(limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(AS_OF_DEFAULT.date()))
    parser.add_argument("--artifact", required=True, help="artifacts/YYYYMMDD")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", default="tonight.csv")
    args = parser.parse_args()
    table = score_batch(args.as_of, Path(args.artifact), limit=args.limit)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}  n={len(table)}  version={table['model_version'].iloc[0] if len(table) else '?'}")


if __name__ == "__main__":
    main()
