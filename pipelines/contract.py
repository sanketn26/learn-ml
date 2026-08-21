"""The scoring contract. Training writes this; the handler only reads it."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipelines.features import FEATURE_COLS

NUMBER = (int, float, np.integer, np.floating)
REQUIRED = {
    "mrr": NUMBER,
    "tenure_so_far": NUMBER,
    "log_usage": NUMBER,
    "features_adopted": NUMBER,
    "total_events": NUMBER,
    "n_support": NUMBER,
    "plan_type": (str,),
}
PLANS = {"free", "starter", "pro", "enterprise"}


def validate(payload: dict) -> None:
    missing = [k for k in REQUIRED if k not in payload]
    extra = [k for k in payload if k not in REQUIRED]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if extra:
        raise ValueError(f"unknown keys: {extra}")
    for key, types in REQUIRED.items():
        if not isinstance(payload[key], types):
            raise TypeError(f"{key} should be {types}, got {type(payload[key])}")
        if key != "plan_type" and pd.isna(payload[key]):
            raise ValueError(f"{key} is missing")
    if payload["plan_type"] not in PLANS:
        raise ValueError(f"unknown plan_type {payload['plan_type']}")


def load_artifact(path: Path) -> dict:
    path = Path(path)
    bundle = joblib.load(path / "model.joblib")
    meta = json.loads((path / "metrics.json").read_text())
    return {**bundle, "metrics": meta, "dir": path}


def predict(payload: dict, artifact: dict, threshold: float | None = None) -> dict:
    validate(payload)
    cut = artifact["metrics"]["threshold"] if threshold is None else threshold
    row = pd.DataFrame([payload])[FEATURE_COLS]
    score = float(artifact["pipeline"].predict_proba(row)[0, 1])
    return {
        "churn_score": round(score, 4),
        "flag_for_cs": score >= cut,
        "model_version": artifact["metrics"]["model_version"],
    }
