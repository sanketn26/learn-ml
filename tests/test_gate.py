from __future__ import annotations

import json
from pathlib import Path

from pipelines.promote import gate


def _metrics(dir: Path, **vals):
    dir.mkdir(parents=True)
    payload = {
        "pr_auc": 0.20,
        "dummy_pr_auc": 0.08,
        "auc": 0.70,
        **vals,
    }
    (dir / "metrics.json").write_text(json.dumps(payload))


def test_refuses_worse_than_dummy(tmp_path: Path):
    cand = tmp_path / "cand"
    _metrics(cand, pr_auc=0.05, dummy_pr_auc=0.08)
    ok, reason = gate(cand, None)
    assert not ok
    assert "dummy" in reason


def test_refuses_to_replace_a_better_prod(tmp_path: Path):
    cand = tmp_path / "cand"
    prod = tmp_path / "prod"
    _metrics(cand, pr_auc=0.15)
    _metrics(prod, pr_auc=0.25)
    ok, _ = gate(cand, prod)
    assert not ok


def test_promotes_first_model_that_beats_dummy(tmp_path: Path):
    cand = tmp_path / "cand"
    _metrics(cand)
    ok, reason = gate(cand, None)
    assert ok
    assert reason == "ok"


def test_refuses_to_compare_models_that_answer_different_questions(tmp_path: Path):
    cand = tmp_path / "cand"
    prod = tmp_path / "prod"
    _metrics(cand, pr_auc=0.05, dummy_pr_auc=0.02, horizon_days=30)
    _metrics(prod, pr_auc=0.13, dummy_pr_auc=0.05, horizon_days=90)
    ok, reason = gate(cand, prod)
    assert not ok
    assert "different question" in reason
