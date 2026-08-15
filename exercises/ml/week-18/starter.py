"""Week 18 exercises — Week 18 — CNNs: Sliding Detectors.

Run from the repo root:

    python exercises/ml/week-18/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()


def main() -> None:
    print(f"data: {DATA}")
    print("Customer 360 sample rows:", len(load_customer_360(n=20)))
    # TODO: implement the tasks in README.md


if __name__ == "__main__":
    main()
