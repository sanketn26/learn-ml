"""Framework-track smoke: files exist, optional imports, syntax-check snippets.

No API keys. No network. Missing langchain/langgraph/crewai → skip, do not fail.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

ML_EXERCISE_DIRS = [ROOT / "exercises" / "ml" / f"week-{i:02d}" for i in range(0, 21)] + [
    ROOT / "exercises" / "ml" / "capstone"
]
ML_DOCS = [ROOT / "docs" / "ml" / "exercises" / f"week-{i:02d}.md" for i in range(0, 21)] + [
    ROOT / "docs" / "ml" / "exercises" / "capstone.md"
]
ML_SECTIONS = (
    "What you are building",
    "Predict before you run",
    "Task",
    "Success criteria",
    "Debugging clues",
    "After you run",
    "Lesson link",
)

LC = [ROOT / "docs" / "langchain" / "exercises" / f"week-{i:02d}.md" for i in range(1, 8)]
LG = [ROOT / "docs" / "langgraph" / "exercises" / f"week-{i:02d}.md" for i in range(1, 6)]
CA = [ROOT / "docs" / "crewai" / "exercises" / f"week-{i:02d}.md" for i in range(1, 5)]
REQUIRED_SECTIONS = (
    "Predict before you run",
    "Runnable command",
    "Expected observation",
    "Self-check",
)


def test_ml_exercise_readmes_have_standard_sections():
    missing = []
    for folder in ML_EXERCISE_DIRS:
        path = folder / "README.md"
        if not path.exists():
            missing.append(str(path))
            continue
        text = path.read_text()
        for heading in ML_SECTIONS:
            if heading not in text:
                missing.append(f"{path.relative_to(ROOT)}: {heading}")
    for path in ML_DOCS:
        text = path.read_text()
        for heading in ML_SECTIONS:
            if heading not in text:
                missing.append(f"{path.relative_to(ROOT)}: {heading}")
    assert not missing, missing


def test_framework_exercise_pages_exist():
    missing = [p for p in LC + LG + CA if not p.exists()]
    assert not missing, f"missing exercise pages: {missing}"


def test_framework_exercise_pages_have_recovery_sections():
    missing = []
    for path in LC + LG + CA:
        text = path.read_text()
        for heading in REQUIRED_SECTIONS:
            if heading not in text:
                missing.append(f"{path.name}: {heading}")
    assert not missing, missing


@pytest.mark.parametrize(
    "mod",
    [
        "langchain_core",
        "langgraph",
        "crewai",
    ],
)
def test_optional_framework_import(mod: str):
    pytest.importorskip(mod)


def _python_files_under_exercises() -> list[Path]:
    found = []
    for track in ("langchain", "langgraph", "crewai"):
        base = ROOT / "exercises" / track
        if not base.exists():
            continue
        found.extend(base.rglob("*.py"))
    return found


def test_syntax_check_framework_python_if_present():
    files = _python_files_under_exercises()
    if not files:
        pytest.skip("no Python snippets under exercises/{langchain,langgraph,crewai}")
    for path in files:
        py_compile.compile(str(path), doraise=True)


def test_fenced_python_in_framework_exercises_parses_or_is_sketch():
    """Best-effort: full-module fences must parse. Incomplete sketches are allowed."""
    errors = []
    for path in LC + LG + CA:
        text = path.read_text()
        chunks = text.split("```")
        for i in range(1, len(chunks), 2):
            header, _, body = chunks[i].partition("\n")
            lang = header.strip().split()[0].lower() if header.strip() else ""
            if lang not in {"python", "py"}:
                continue
            src = body.strip()
            if not src or src.startswith("#") and "TODO" in src:
                continue
            try:
                ast.parse(src)
            except SyntaxError:
                # snippets in these pages are often incomplete on purpose
                if "..." in src or src.lstrip().startswith("app.") or "TODO" in src:
                    continue
                errors.append(f"{path.name}: failed to parse a python fence")
    assert not errors, errors
