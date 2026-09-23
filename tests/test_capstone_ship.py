"""Gate for the job-path capstone (docs/ml/capstone-ship.md)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone_ship.briefs import RETENTION_DESK, judge, select, threshold_for
from capstone_ship.incident import DEFECTS, _defect_for, diagnose, incident_frame
from pipelines.contract import validate
from pipelines.features import FEATURE_COLS, NUMERIC, build_features
from pipelines.score_batch import _payload

ROOT = Path(__file__).resolve().parent.parent
NIGHT = pd.Timestamp("2024-06-15")


@pytest.fixture(scope="module")
def clean():
    return build_features(as_of=NIGHT, n=3000)


def test_seeds_reach_every_defect():
    assert {_defect_for(seed) for seed in range(30)} == set(DEFECTS)


@pytest.mark.parametrize("name", sorted(DEFECTS))
def test_defect_slips_past_the_contract_and_touches_only_its_columns(clean, name):
    apply, corrupted = DEFECTS[name]
    bad = apply(clean, NIGHT)
    for rec in bad[FEATURE_COLS].to_dict(orient="records"):
        validate(_payload(rec))  # the whole point: the contract cannot see it
    changed = {c for c in NUMERIC if not np.allclose(bad[c].astype(float), clean[c].astype(float))}
    assert changed == set(corrupted)


def test_incident_frame_is_deterministic():
    a = incident_frame(NIGHT, seed=3, n=500)
    b = incident_frame(NIGHT, seed=3, n=500)
    pd.testing.assert_frame_equal(a, b)


def test_diagnose_wants_the_exact_columns():
    for seed in range(6):
        _, corrupted = DEFECTS[_defect_for(seed)]
        assert diagnose(seed, set(corrupted))
        assert not diagnose(seed, set(corrupted) | {"plan_type"})
        assert not diagnose(seed, set())


def test_select_cuts_at_capacity_and_threshold_reproduces_it(clean):
    scores = np.linspace(0, 1, len(clean))
    picked = select(clean, scores, RETENTION_DESK)
    assert len(picked) == RETENTION_DESK.capacity
    assert (scores >= threshold_for(picked)).sum() == RETENTION_DESK.capacity
    y = pd.Series(0, index=clean.index)
    y.loc[picked.index[:4]] = 1
    verdict = judge(picked, clean, y, RETENTION_DESK)
    assert verdict["hits"] == 4 and verdict["recall"] == 1.0


def _solution():
    path = ROOT / "solutions" / "ml" / "capstone-ship" / "solution.py"
    spec = importlib.util.spec_from_file_location("capstone_ship_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_solution_ships_and_diagnoses(tmp_path: Path):
    solution = _solution()
    seeds = tuple(sorted({_defect_for(s): s for s in range(30)}.values()))  # one seed per defect
    result = solution.run(tmp_path, seeds=seeds, n=4000)

    metrics = json.loads((tmp_path / "artifacts" / "prod" / "metrics.json").read_text())
    assert metrics["pr_auc"] > metrics["dummy_pr_auc"]  # the stable signal; precision@80 is 0–2 hits of noise
    assert metrics["brief"] == RETENTION_DESK.key and metrics["capacity"] == RETENTION_DESK.capacity
    assert len(result["tonight"]) == RETENTION_DESK.capacity
    for seed, incident in result["incidents"].items():
        assert incident["diagnosed"], (seed, _defect_for(seed), incident["suspects"])
        assert incident["names_changed"] > 0
