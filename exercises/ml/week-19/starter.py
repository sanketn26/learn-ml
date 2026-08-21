"""Week 19 exercises — Week 19 — RNNs: A Clipboard That Walks the Sequence.

Run from the repo root:

    python exercises/ml/week-19/starter.py
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
    print("usage grid", X.shape, "lifetime churn rate", float(y.mean()), "(sequence toy)")
    print("TODO: paste SequenceNet from the lesson, then try GRU vs RNN")


if __name__ == "__main__":
    main()
