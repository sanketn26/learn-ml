"""Week 19 — run the job, do not overwrite prod from train.

    python exercises/ml/week-19/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.train import train


def main() -> None:
    meta = train("2024-06-01", ROOT / "artifacts", n=4000)
    print(meta)
    print("next: python -m pipelines.promote --candidate artifacts/20240601")
    print("then: python -m pipelines.score_batch --artifact artifacts/prod --out tonight.csv")


if __name__ == "__main__":
    main()
