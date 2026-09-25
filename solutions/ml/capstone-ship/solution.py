"""Job-path capstone reference solution — ship the churn score.

Run from the repo root:

    python solutions/ml/capstone-ship/solution.py

Writes under artifacts/capstone-ship/ (gitignored). Every step reuses
pipelines/ — the capstone is the composition plus two decisions the weeks
never made for you: a stated capacity budget, and a diagnosis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone_ship.briefs import RETENTION_DESK, Brief, judge, select, threshold_for
from capstone_ship.incident import column_shift, diagnose, incident_frame
from pipelines.contract import load_artifact, predict, validate
from pipelines.features import FEATURE_COLS, NUMERIC, build_features
from pipelines.promote import gate, promote
from pipelines.score_batch import _payload, score_batch
from pipelines.split import snapshot_split
from pipelines.train import train

AS_OF = pd.Timestamp("2024-06-01")
HORIZON = 30
INCIDENT_NIGHT = AS_OF + pd.Timedelta(days=14)


def step1_features(as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    frame = build_features(as_of=as_of, n=None)
    assert frame["user_id"].is_unique, "grain: one row per at-risk customer"
    assert frame["signup_date"].max() <= as_of, "nobody from the future"
    assert "tenure_days" not in FEATURE_COLS and "churn_date" not in FEATURE_COLS
    return frame


def step2_labels(as_of: pd.Timestamp = AS_OF, horizon: int = HORIZON):
    train_df, y_train, test_df, y_test = snapshot_split(as_of, horizon_days=horizon)
    assert y_train.sum() > 0 and y_test.sum() > 0, "a horizon with no positives cannot be supervised"
    return train_df, y_train, test_df, y_test


def step3_train(out_dir: Path, as_of: pd.Timestamp = AS_OF, horizon: int = HORIZON, n: int | None = None) -> Path:
    meta = train(str(as_of.date()), out_dir, n=n, horizon_days=horizon)
    candidate = out_dir / meta["model_version"]
    ok, reason = gate(candidate, None)
    assert ok, f"candidate loses to the dummy: {reason}"
    return candidate


def step4_threshold(candidate: Path, test_df: pd.DataFrame, y_test: pd.Series, brief: Brief = RETENTION_DESK) -> dict:
    art = load_artifact(candidate)
    scores = art["pipeline"].predict_proba(test_df[FEATURE_COLS])[:, 1]
    picked = select(test_df, scores, brief)
    verdict = judge(picked, test_df, y_test, brief)
    verdict["threshold"] = threshold_for(picked)
    verdict["flagged_at_0.5"] = int((scores >= 0.5).sum())

    metrics_path = candidate / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update({"threshold": round(verdict["threshold"], 4), "brief": brief.key, "capacity": brief.capacity})
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return verdict


def step5_contract(candidate: Path, test_df: pd.DataFrame) -> dict:
    art = load_artifact(candidate)
    payload = _payload(test_df[FEATURE_COLS].iloc[0].to_dict())
    response = predict(payload, art)
    assert set(response) == {"churn_score", "flag_for_cs", "model_version"}
    for bad, why in (
        ({**payload, "user_id": "user_041906"}, "keys are not features"),
        ({**payload, "mrr": float("nan")}, "NaN is missing, not zero"),
        ({k: v for k, v in payload.items() if k != "plan_type"}, "every feature is required"),
    ):
        try:
            validate(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"validate accepted a bad payload: {why}")
    return response


def step6_promote_and_score(candidate: Path, prod: Path, as_of: pd.Timestamp = AS_OF,
                            brief: Brief = RETENTION_DESK) -> pd.DataFrame:
    promote(candidate, prod)
    tonight = score_batch(str(as_of.date()), prod, limit=brief.capacity)
    tonight.to_csv(prod.parent / "tonight.csv", index=False)
    return tonight


CRON = """set -euo pipefail
python -m pytest tests/
python -m pipelines.train --as-of "$AS_OF"
python -m pipelines.promote --candidate "artifacts/${AS_OF//-/}"
python -m pipelines.score_batch --as-of "$AS_OF" --artifact artifacts/prod --limit 80 --out tonight.csv
"""


def step7_incident(prod: Path, seed: int, night: pd.Timestamp = INCIDENT_NIGHT,
                   reference: pd.DataFrame | None = None) -> dict:
    art = load_artifact(prod)
    last_week = reference if reference is not None else build_features(night - pd.Timedelta(days=7), n=None)
    clean = build_features(night, n=None)
    bad = incident_frame(night, seed=seed)

    shift = column_shift(last_week, bad, NUMERIC)
    moved = shift[((shift["mean_ratio"] - 1).abs() > 0.10)
                  | ((shift["tonight_zero_share"] - shift["ref_zero_share"]).abs() > 0.01)]
    suspects = sorted(moved.index)

    def top(frame: pd.DataFrame) -> set:
        scores = art["pipeline"].predict_proba(frame[FEATURE_COLS])[:, 1]
        return set(frame["user_id"].to_numpy()[np.argsort(-scores)[:RETENTION_DESK.capacity]])

    return {
        "suspects": suspects,
        "diagnosed": diagnose(seed, suspects),
        "names_changed": RETENTION_DESK.capacity - len(top(clean) & top(bad)),
        "shift": shift.round(3),
    }


def run(workdir: Path, seeds: tuple[int, ...] = (0,), n: int | None = None) -> dict:
    out = workdir / "artifacts"
    prod = out / "prod"
    step1_features()
    _, _, test_df, y_test = step2_labels()
    candidate = step3_train(out, n=n)
    verdict = step4_threshold(candidate, test_df, y_test)
    response = step5_contract(candidate, test_df)
    tonight = step6_promote_and_score(candidate, prod)
    reference = build_features(INCIDENT_NIGHT - pd.Timedelta(days=7), n=None)
    incidents = {seed: step7_incident(prod, seed, reference=reference) for seed in seeds}
    return {"verdict": verdict, "response": response, "tonight": tonight, "incidents": incidents}


def main() -> None:
    result = run(ROOT / "artifacts" / "capstone-ship", seeds=(0,))
    v = result["verdict"]
    print("4. brief:", v["brief"], f"precision@{v['picked']}={v['precision']:.3f} (base rate {v['base_rate']:.4f}),",
          f"recall={v['recall']:.3f}, threshold={v['threshold']:.4f}, flagged at 0.5={v['flagged_at_0.5']}")
    print("5. contract:", result["response"])
    print("6. tonight.csv rows:", len(result["tonight"]))
    print("   cron:\n" + "\n".join("     " + line for line in CRON.strip().splitlines()))
    for seed, inc in result["incidents"].items():
        print(f"7. seed {seed}: suspects={inc['suspects']} diagnosed={inc['diagnosed']} "
              f"names changed on Priya's list={inc['names_changed']}/80")
        print(inc["shift"][["mean_ratio", "ref_zero_share", "tonight_zero_share"]].to_string())


if __name__ == "__main__":
    main()
