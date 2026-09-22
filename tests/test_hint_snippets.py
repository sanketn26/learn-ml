"""Every "Hint 3" python block on an exercise page must run.

Hint 3 is near-complete code. A snippet that no longer matches the
pipelines/ API is worse than no hint. The blocks on one page run top to
bottom in one namespace (later tasks reuse earlier names), in a scratch
directory so nothing lands in the repo's artifacts/.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "docs" / "ml" / "exercises").glob("*.md"))

HINT3 = re.compile(r'^\?\?\? example "Hint 3[^"]*"\n((?:[ ]{4}.*\n|\n)+)', re.M)
PYBLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def hint3_code(page: Path) -> str:
    chunks = []
    for body in HINT3.findall(page.read_text()):
        for code in PYBLOCK.findall(textwrap.dedent(body)):
            chunks.append(code)
    return "\n".join(chunks)


@pytest.mark.parametrize("page", [p for p in PAGES if hint3_code(p)], ids=lambda p: p.stem)
def test_hint3_snippets_run(page: Path, tmp_path: Path):
    script = tmp_path / f"{page.stem}_hint3.py"
    script.write_text(hint3_code(page))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"{page.name} Hint 3 failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
