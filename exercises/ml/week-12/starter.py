"""Week 12 capstone — time split, predict() contract, capacity threshold.

Run from the repo root:

    python exercises/ml/week-12/starter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import load_customer_360


def main() -> None:
    df = load_customer_360()
    print(f"rows={len(df):,}  columns={list(df.columns)}")
    print("TODO 1: time-based split on signup_date vs a shuffled split")
    print("TODO 2: validate() + predict() with a version string")
    print("TODO 3: threshold that flags at most 80 customers")
    print("TODO 4: overlay train vs later-signup histograms")
    print("TODO 5: one-page write-up in this folder as WRITEUP.md")


if __name__ == "__main__":
    main()
