"""Week 20 exercises — Week 20 — Transformers: Everything Looks at Everything.

Run from the repo root:

    python exercises/ml/week-20/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir

DATA = find_data_dir()


def main() -> None:
    print(f"data: {DATA}")
    print("TODO: paste the 3-token self-attention block from the lesson")
    print("TODO: classify feedback.json with character ids (category matches the text)")


if __name__ == "__main__":
    main()
