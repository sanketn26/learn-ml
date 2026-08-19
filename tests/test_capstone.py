import pytest

from capstone.evaluate import compare
from capstone.reliability import RejectedCall, validate_call
from capstone.scenarios import SCENARIOS
from capstone.teacher import build_trajectory


def test_every_golden_trajectory_passes_its_own_contract():
    for scenario in SCENARIOS:
        build_trajectory(scenario)  # raises RejectedCall internally if invalid


def test_unknown_tool_is_rejected():
    with pytest.raises(RejectedCall):
        validate_call({"name": "delete_repo", "arguments": {}})


def test_missing_required_arg_is_rejected():
    with pytest.raises(RejectedCall):
        validate_call({"name": "explain_error", "arguments": {}})


def test_valid_call_passes():
    call = {"name": "explain_error", "arguments": {"traceback": "Traceback..."}}
    assert validate_call(call) == call


def test_specialist_placeholder_beats_general_baseline():
    specialist, baseline = compare()
    assert specialist["accuracy"] == 1.0
    assert specialist["accuracy"] > baseline["accuracy"]
