"""Week 18 exercises — Week 18 — CNNs: Sliding Detectors.

Run from the repo root:

    python exercises/ml/week-18/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir, load_weekly_usage_grid

DATA = find_data_dir()


def main() -> None:
    print(f"data: {DATA}")
    X, y = load_weekly_usage_grid(n_users=500)
    print("usage grid", X.shape, "churn rate", float(y.mean()))
    print("TODO: paste UsageCNN from the lesson, then change kernel_size")


if __name__ == "__main__":
    main()
