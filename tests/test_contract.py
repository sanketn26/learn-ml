from __future__ import annotations

import pytest

from pipelines.contract import validate


def _ok(**overrides):
    base = {
        "mrr": 29.0,
        "tenure_so_far": 80,
        "log_usage": 1.2,
        "features_adopted": 3,
        "total_events": 10,
        "n_support": 0,
        "plan_type": "starter",
    }
    base.update(overrides)
    return base


def test_validate_accepts_a_legal_payload():
    validate(_ok())


def test_validate_rejects_missing_and_extra():
    with pytest.raises(ValueError, match="missing"):
        validate({"mrr": 1.0})
    with pytest.raises(ValueError, match="unknown"):
        validate(_ok(churn_date="2024-07-01"))


def test_validate_rejects_pii_shaped_fields():
    with pytest.raises(ValueError, match="unknown"):
        validate(_ok(email="ada@cloudwave.test"))
