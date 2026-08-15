"""Week 17 — golden router + a read-only score tool.

    python exercises/ml/week-17/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.router import evaluate


def main() -> None:
    print("golden failures", evaluate())
    print("TODO: get_churn_score against artifacts/prod")
    print("TODO: incident write-up")


if __name__ == "__main__":
    main()
