"""Regenerate exercises/ml/<week>/README.md from docs/ml/exercises/<week>.md.

The docs page is the source of truth; the README is the copy a learner reads
next to starter.py (and on GitHub). MkDocs `??? kind "title"` collapsibles
become <details> blocks, which GitHub also renders closed.

    python scripts/sync_exercise_readmes.py          # write
    python scripts/sync_exercise_readmes.py --check  # exit 1 if any README is stale
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "ml" / "exercises"
EXERCISES = ROOT / "exercises" / "ml"

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
COLLAPSIBLE = re.compile(r'^\?\?\?\+? [a-z]+ "([^"]*)"\n((?:[ ]{4}.*\n|\n)+)', re.M)


def _details(match: re.Match) -> str:
    title, body = match.group(1), textwrap.dedent(match.group(2)).strip("\n")
    return f"<details>\n<summary>{title}</summary>\n\n{body}\n\n</details>\n\n"


def render(page: Path) -> str:
    text = FRONT_MATTER.sub("", page.read_text()).lstrip("\n")
    text = text.replace("# Exercises — ", "# Exercise — ", 1)
    text = COLLAPSIBLE.sub(_details, text)
    text = text.replace("](../", "](../../../docs/ml/")
    return re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n"


def pairs() -> list[tuple[Path, Path]]:
    out = []
    for page in sorted(DOCS.glob("*.md")):
        readme = EXERCISES / page.stem / "README.md"
        if readme.parent.is_dir():
            out.append((page, readme))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for page, readme in pairs():
        want = render(page)
        if readme.exists() and readme.read_text() == want:
            continue
        stale.append(readme.relative_to(ROOT))
        if not args.check:
            readme.write_text(want)
    verb = "stale" if args.check else "wrote"
    for path in stale:
        print(f"{verb}: {path}")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
