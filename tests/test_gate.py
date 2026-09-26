from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipelines.features import FEATURE_COLS, make_preprocessor
from pipelines.promote import gate


def _holdout(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS if c != "plan_type"})
    frame["plan_type"] = rng.choice(["free", "starter", "pro"], size=n)
    y = pd.Series((frame["log_usage"] + 0.3 * rng.normal(size=n) < -1).astype(int))
    return frame, y


def _artifact(dir: Path, holdout, signal: str | None = "log_usage", **vals) -> None:
    """A real (tiny) pipeline plus metrics.json. signal=None trains on noise."""
    frame, y = holdout
    target = y if signal else pd.Series(np.random.default_rng(1).integers(0, 2, len(y)))
    pipe = Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression())]).fit(frame[FEATURE_COLS], target)
    dir.mkdir(parents=True)
    joblib.dump({"pipeline": pipe, "features": FEATURE_COLS}, dir / "model.joblib")
    payload = {"pr_auc": 0.20, "dummy_pr_auc": 0.08, "auc": 0.70, "horizon_days": 30, **vals}
    (dir / "metrics.json").write_text(json.dumps(payload))


def test_refuses_worse_than_dummy(tmp_path: Path):
    cand = tmp_path / "cand"
    _artifact(cand, _holdout(), pr_auc=0.05, dummy_pr_auc=0.08)
    ok, reason = gate(cand, None)
    assert not ok
    assert "dummy" in reason


def test_promotes_first_model_that_beats_dummy(tmp_path: Path):
    cand = tmp_path / "cand"
    _artifact(cand, _holdout())
    assert gate(cand, None) == (True, "ok")


def test_refuses_to_replace_a_better_prod_on_the_same_holdout(tmp_path: Path):
    holdout = _holdout()
    cand, prod = tmp_path / "cand", tmp_path / "prod"
    _artifact(cand, holdout, signal=None, pr_auc=0.40)   # a great stored number, a noise model
    _artifact(prod, holdout, pr_auc=0.10)                # a modest stored number, a real model
    ok, reason = gate(cand, prod, holdout=holdout)
    assert not ok
    assert "same holdout" in reason


def test_stored_numbers_from_different_months_do_not_decide(tmp_path: Path):
    # prod's metrics.json says 0.60 — measured on its own month, at a higher base rate.
    # On the candidate's holdout the candidate is at least as good, so it promotes.
    holdout = _holdout()
    cand, prod = tmp_path / "cand", tmp_path / "prod"
    _artifact(cand, holdout, pr_auc=0.15)
    _artifact(prod, holdout, signal=None, pr_auc=0.60)
    ok, reason = gate(cand, prod, holdout=holdout)
    assert ok, reason


def test_refuses_to_compare_models_that_answer_different_questions(tmp_path: Path):
    holdout = _holdout()
    cand, prod = tmp_path / "cand", tmp_path / "prod"
    _artifact(cand, holdout, pr_auc=0.05, dummy_pr_auc=0.02, horizon_days=30)
    _artifact(prod, holdout, pr_auc=0.13, dummy_pr_auc=0.05, horizon_days=90)
    ok, reason = gate(cand, prod, holdout=holdout)
    assert not ok
    assert "different question" in reason
