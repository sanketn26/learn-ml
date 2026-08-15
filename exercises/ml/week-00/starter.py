"""Week 00 exercises — Week 0 — Strong Python for AI Engineers.

Run from the repo root:

    python exercises/ml/week-00/starter.py
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
    print("TODO 1: churn rate per plan_type with csv + Counter")
    print("TODO 2: CustomerFeatures.to_payload()")
    print("TODO 3: MeanBaseline fit/predict asserts")
    print("TODO 4: mutable-default foot-gun, then the fix")


if __name__ == "__main__":
    main()
