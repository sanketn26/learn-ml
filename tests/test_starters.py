"""Every ML starter runs untouched, and its checks accept a correct answer.

A starter with no answers must report "not started" for every task, never
crash. With the reference answers in tests/starter_refs/ patched in, every
check must pass — otherwise the check, not the learner, is wrong.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STARTERS = sorted((ROOT / "exercises" / "ml").glob("week-*/starter.py"))
REFS = {p.stem: p for p in (ROOT / "tests" / "starter_refs").glob("week_*.py")}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"starter_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_checks(paths):
    return [p for p in paths if hasattr(_load(p), "CHECKS")]


@pytest.mark.parametrize("path", _with_checks(STARTERS), ids=lambda p: p.parent.name)
def test_untouched_starter_reports_not_started(path: Path, capsys):
    module = _load(path)
    module.run(module.CHECKS)
    out = capsys.readouterr().out
    assert "✗" not in out, out
    assert "✓" not in out, out


@pytest.mark.parametrize(
    "path",
    [p for p in _with_checks(STARTERS) if p.parent.name.replace("-", "_") in REFS],
    ids=lambda p: p.parent.name,
)
def test_reference_answers_pass_every_check(path: Path, capsys):
    module = _load(path)
    ref = importlib.import_module(f"tests.starter_refs.{path.parent.name.replace('-', '_')}")
    for i, task in enumerate(module.CHECKS, 1):
        if task.solve is not None and not task.writeup:
            task.solve = getattr(ref, task.solve.__name__)
    ok = module.run(module.CHECKS)
    assert ok, capsys.readouterr().out
