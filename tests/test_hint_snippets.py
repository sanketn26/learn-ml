"""Every "Hint 3" python block on an exercise page must run.

Hint 3 is near-complete code. A snippet that no longer matches the
pipelines/ or framework API is worse than no hint. The blocks on one page
run top to bottom in one namespace (later tasks reuse earlier names), in a
scratch directory so nothing lands in the repo's artifacts/.

Framework pages need their own venv (make setup-frameworks / setup-crewai).
A page whose imports are missing here is skipped, not failed — run
`.venv-framework/bin/python -m pytest tests/test_hint_snippets.py` to cover them.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(
    p for track in ("ml", "langchain", "langgraph", "crewai") for p in (DOCS / track / "exercises").glob("*.md")
)

HINT3 = re.compile(r'^\?\?\? example "Hint 3[^"]*"\n((?:[ ]{4}.*\n|\n)+)', re.M)
PYBLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def hint3_code(page: Path) -> str:
    chunks = []
    for body in HINT3.findall(page.read_text()):
        for code in PYBLOCK.findall(textwrap.dedent(body)):
            chunks.append(code)
    return "\n".join(chunks)


def missing_modules(code: str) -> list[str]:
    roots = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    local = {"lib", "pipelines", "capstone", "eval"}
    return sorted(m for m in roots - local if importlib.util.find_spec(m) is None)


def _id(page: Path) -> str:
    return f"{page.parent.parent.name}-{page.stem}"


@pytest.mark.parametrize("page", [p for p in PAGES if hint3_code(p)], ids=_id)
def test_hint3_snippets_run(page: Path, tmp_path: Path):
    code = hint3_code(page)
    missing = missing_modules(code)
    if missing:
        pytest.skip(f"not installed in this venv: {missing}")
    script = tmp_path / f"{_id(page)}_hint3.py"
    script.write_text(code)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    rel = page.relative_to(ROOT)
    assert proc.returncode == 0, f"{rel} Hint 3 failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
