"""Gate for the agent capstone (docs/ml/capstone-agent.md).

The parts (runbooks, ledger) are stdlib and always run. The graph needs the
framework venv: `.venv-framework/bin/python -m pytest tests/test_capstone_agent.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from capstone_agent.golden import GOLDEN, SCORES
from capstone_agent.ledger import Ledger, ProcessDied
from capstone_agent.runbooks import retrieve

ROOT = Path(__file__).resolve().parent.parent


def test_retrieve_answers_known_questions_and_refuses_the_open_incident():
    assert retrieve("How do I get an API key?")[0][1] == "api-keys"
    assert retrieve("My export keeps timing out around 150k rows (CW-1847). Why?") == []


def test_ledger_is_idempotent_even_across_a_crash():
    ledger = Ledger()
    ledger.crash_after_next_write()
    with pytest.raises(ProcessDied):
        ledger.credit("t1:CW-1847:credit", "user_041906", 2900)
    replay = ledger.credit("t1:CW-1847:credit", "user_041906", 2900)
    assert replay["replayed"] and ledger.total_cents("user_041906") == 2900


def test_an_unkeyed_write_pays_twice():
    ledger = Ledger()
    ledger.credit("attempt-1", "user_041906", 2900)
    ledger.credit("attempt-2", "user_041906", 2900)  # what a timestamp or uuid key does on resume
    assert ledger.total_cents("user_041906") == 5800


def _solution():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    path = ROOT / "solutions" / "ml" / "capstone-agent" / "solution.py"
    spec = importlib.util.spec_from_file_location("capstone_agent_solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # LangGraph resolves TypedDict hints through sys.modules
    spec.loader.exec_module(module)
    return module


def test_reference_agent_passes_every_golden_ticket():
    from capstone_agent.golden import evaluate

    rows = evaluate(_solution().build_agent)
    assert [r for r in rows if not r["ok"]] == []
    assert len(rows) == len(GOLDEN)


def test_golden_gate_catches_an_agent_without_the_approval_interrupt():
    from capstone_agent.golden import evaluate

    solution = _solution()
    rows = evaluate(lambda ledger, scores: solution.build_agent(ledger, scores, approval_gate=False))
    failed = {r["id"] for r in rows if not r["ok"]}
    assert "g5" in failed


def test_decisions_and_crash_drills():
    solution = _solution()
    decisions = solution.drill_decisions()
    assert decisions["outcomes"] == {"approve": "credited", "reject": "cancelled", "needs_info": "asked-for-info"}
    assert decisions["credited_cents"] == solution.CREDIT_CENTS
    crash = solution.drill_crash_after_write()
    assert crash["last_log"] == "replayed" and crash["ledger_calls"] == 2
    assert crash["credited_cents"] == solution.CREDIT_CENTS
    assert SCORES  # fixture present
