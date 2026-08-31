"""Capstone reference — Phases 1–2 and 4–5. No trained weights.

Run from the repo root:

    python solutions/ml/capstone/solution.py

Phase 3: capstone/finetune/ (dry-run on CPU; real train on Colab/GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone.evaluate import compare, score_one
from capstone.reliability import RejectedCall, validate_call
from capstone.scenarios import SCENARIOS, Scenario
from capstone.teacher import build_trajectory, golden_call, write_splits
from capstone.tools import TOOLS, tool_schema


def extra_scenario() -> Scenario:
    """Exercise 2 — a seventh scenario, local to this script (do not widen TOOLS)."""
    return Scenario(
        id="s7_style_init",
        expect_tool="check_style_or_conventions",
        context="Another PEP-8 nit: __init__ named as a factory.",
        input_text="def InitModel(cfg):\n    return GradientBoostingClassifier(**cfg)\n",
    )


def main() -> None:
    print("Phase 1 — tool contract")
    print("  tools:", [t["name"] for t in tool_schema()])
    print("  required example:", {name: spec["required"] for name, spec in TOOLS.items()})

    print("\nExercise 1 — break the contract on purpose")
    for call in (
        {"name": "delete_repo", "arguments": {}},
        {"name": "explain_error", "arguments": {}},
    ):
        try:
            validate_call(call)
        except RejectedCall as exc:
            print(f"  {call['name']!r}: {exc}")

    print("\nPhase 2 — teacher trajectories (correct by construction)")
    counts = write_splits(ROOT / "capstone" / "data")
    print("  splits:", counts)
    for scenario in SCENARIOS:
        row = build_trajectory(scenario)
        assert row["tool_call"] is None or row["tool_call"]["name"] in TOOLS

    print("\nExercise 2 — seventh scenario, locally")
    s7 = extra_scenario()
    traj = build_trajectory(s7)
    print("  golden:", traj["tool_call"])
    print("  score_one:", score_one(s7, golden_call(s7)))
    print("  placeholder specialist stays at 1.0 because validate_call runs inside build_trajectory")

    print("\nExercise 3 — ship / don't ship vs a 0.75 real model")
    specialist, baseline = compare()
    print("  specialist (placeholder ceiling):", specialist)
    print("  baseline:", baseline)
    print(
        "  If a *real* adapter landed at 0.75 accuracy: beat the general baseline, but "
        "do not ship destructive tools, and do not treat 0.75 as the placeholder ceiling. "
        "Ship only on narrow, repeated coding tasks behind validate_call(); otherwise route out."
    )

    print("\nExercise 4 — base-model memo")
    print(
        "  Start Phase 3 with FunctionGemma only as a formatting baseline, then rerun the same\n"
        "  recipe on a small Qwen/Phi coding variant. Mind-changer: tool-selection accuracy and\n"
        "  hallucination rate on capstone/scenarios.py (Phase 5), not loss curves. If FunctionGemma\n"
        "  emits well-formed JSON around a wrong diagnosis, pick the larger coder."
    )

    print("\nPhase 3 — not in this file")
    print("  python capstone/finetune/prepare_data.py --dry-run")
    print("  python capstone/finetune/train_lora.py --dry-run")
    print("  python capstone/finetune/evaluate_adapter.py --dry-run")
    print("  Real train: GPU (Colab T4/L4). See capstone/finetune/README.md")


if __name__ == "__main__":
    main()
