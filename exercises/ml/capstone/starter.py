"""Capstone starter — the on-call specialist. Tasks 1–5 run offline, no GPU, no API key.

    python exercises/ml/capstone/starter.py

Fill in the tasks in order. The runner stops at the first TODO and tells you
which one is next. Task 6 (fine-tune) is capstone/finetune/ — see README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone.cases import DEFAULT_NIGHTS, all_cases, incident_cases  # noqa: F401
from capstone.evaluate import ablation, default_entries, table  # noqa: F401
from capstone.harness import run_loop, run_one_shot  # noqa: F401
from capstone.policies import rules_first, rules_planner, rules_policy, teacher_policy  # noqa: F401
from capstone.prompt import render_state  # noqa: F401
from capstone.reliability import RejectedCall, validate_call  # noqa: F401
from capstone.runbook import accepted, next_call  # noqa: F401
from capstone.spec import load_specs
from capstone.teacher import corrupt  # noqa: F401

NIGHT = "2024-07-06"
WINDOW = {"ref_as_of": "2024-06-29", "as_of": NIGHT}


def task1_break_the_spec() -> list[str]:
    """Three well-formed calls that validate_call still rejects. Return the three messages."""
    raise NotImplementedError("task 1: a fake column, an invented user_id, evidence citing a rejected step")


def task2_walk_one_investigation() -> dict:
    """Teacher on the three-defect night (train wording), rules on the same night (test wording).
    Return {'teacher_steps': int, 'rules_outcome': str}."""
    raise NotImplementedError("task 2: run_loop(teacher_policy(case), case) and render_state")


def task3_what_a_rejection_costs() -> dict:
    """A sloppy policy that corrupts its first call. Return solved rate at budget 8 and at no slack."""
    raise NotImplementedError("task 3: wrap teacher_policy, corrupt the first call, vary the budget")


def task4_overfit_the_baseline() -> dict:
    """Patch rules_first with keywords read off the test tickets. Return solved rates before and after."""
    raise NotImplementedError("task 4: patched_first → next_call loop, test vs val")


def task5_ablation() -> str:
    """The two ablation tables (train, test) as text."""
    raise NotImplementedError("task 5: ablation(cases, default_entries()) per split, then table()")


def main() -> None:
    print(f"command surface: {list(load_specs())}")
    try:
        print("\n1.", task1_break_the_spec())
        print("\n2.", task2_walk_one_investigation())
        print("\n3.", task3_what_a_rejection_costs())
        print("\n4.", task4_overfit_the_baseline())
        print("\n5.\n" + task5_ablation())
        print("\n6. fine-tune: python capstone/finetune/prepare_data.py --dry-run, then Colab — see README.md")
    except NotImplementedError as todo:
        print(f"\nnext: {todo}")


if __name__ == "__main__":
    main()
