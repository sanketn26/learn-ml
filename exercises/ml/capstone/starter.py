"""Capstone starter — Phases 1, 2, 4, 5 run offline, no GPU, no API key.

    python exercises/ml/capstone/starter.py

Phase 3 (fine-tune on Colab) is not run here — see README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone.evaluate import compare
from capstone.reliability import RejectedCall, validate_call
from capstone.scenarios import SCENARIOS
from capstone.teacher import write_splits
from capstone.tools import tool_schema


def main() -> None:
    print(f"tool surface ({len(tool_schema())} tools):",
          [t["name"] for t in tool_schema()])

    print(f"\nscenarios: {len(SCENARIOS)}")
    counts = write_splits(ROOT / "capstone" / "data")
    print("wrote trajectory splits to capstone/data/:", counts)

    print("\nreliability check — a call missing a required arg:")
    try:
        validate_call({"name": "explain_error", "arguments": {}})
    except RejectedCall as exc:
        print("  rejected:", exc)

    print("\nPhase 5 — specialist (placeholder ceiling) vs. general-model baseline:")
    specialist, baseline = compare()
    print("  specialist:", specialist)
    print("  baseline:  ", baseline)
    print(f"  specialization gain (placeholder): "
          f"{specialist['accuracy'] - baseline['accuracy']:+.0%} accuracy")

    # TODO: implement exercises 1-4 in README.md


if __name__ == "__main__":
    main()
