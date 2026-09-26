"""Capstone reference — tasks 1–5, offline. No trained weights.

Run from the repo root:

    python solutions/ml/capstone/solution.py

Task 6 (fine-tune): capstone/finetune/ — dry-run on CPU, train on Colab.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone.cases import DEFAULT_NIGHTS, all_cases, incident_cases
from capstone.evaluate import ablation, default_entries, table
from capstone.harness import run_loop
from capstone.policies import rules_first, rules_policy, teacher_policy
from capstone.prompt import render_state
from capstone.reliability import RejectedCall, validate_call
from capstone.runbook import accepted, next_call
from capstone.teacher import corrupt

NIGHT = "2024-07-06"
WINDOW = {"ref_as_of": "2024-06-29", "as_of": NIGHT}


def task1_break_the_spec() -> list[str]:
    state = {
        "ticket": "Tonight's list barely overlaps last week's.",
        "context": {**WINDOW, "capacity": 80},
        "steps": [
            {"n": 1, "command": "check_grain", "args": {"as_of": NIGHT}, "result": {"grain_ok": True}},
            {"n": 2, "call": {"command": "check_grain", "args": {"as_of": "tonight"}}, "rejected": "not a date"},
        ],
    }
    broken = [
        # feature_column: sounds like a feature, is the leaked lifetime label (Week 8).
        {"command": "compare_nights", "args": {"columns": ["tenure_days"], **WINDOW}},
        # user_id: must appear in an earlier result — the model can't invent a fixture customer.
        {"command": "inspect_customer", "args": {"user_id": "user_000417", **WINDOW}},
        # steps: evidence must be accepted steps; step 2 was rejected.
        {"command": "conclude", "args": {"causes": ["join_fanout"], "columns": ["total_events"], "evidence": [2],
                                         "checks_to_add": ["mean_ratio_vs_last_week"]}},
    ]
    messages = []
    for call in broken:
        try:
            validate_call(call, state)
        except RejectedCall as exc:
            messages.append(str(exc))
    return messages


def _three_defect_night():
    three = [c for c in incident_cases((NIGHT,)) if len(c.defects) == 3]
    return next(c for c in three if c.split == "train"), next(c for c in three if c.split == "test")


def task2_walk_one_investigation() -> dict:
    familiar, unseen = _three_defect_night()
    run = run_loop(teacher_policy(familiar), familiar)
    print(render_state(run["state"]))
    rules = run_loop(rules_policy, unseen)
    return {"teacher_steps": run["accepted"], "rules_outcome": rules["outcome"],
            "rules_first_step": rules["state"]["steps"][0].get("command")}


def task3_what_a_rejection_costs() -> dict:
    def sloppy(case):
        teacher = teacher_policy(case)
        return lambda state: corrupt(teacher(state)) if not state["steps"] else teacher(state)

    cases = [c for c in incident_cases((NIGHT,)) if c.split == "train" and c.defects]
    horizon = {c.id: run_loop(teacher_policy(c), c)["accepted"] for c in cases}
    out = {}
    for label, budget in (("budget_8", lambda c: 8), ("no_slack", lambda c: horizon[c.id])):
        runs = [run_loop(sloppy(c), c, budget=budget(c)) for c in cases]
        out[label] = {"solved": sum(r["outcome"] == "solved" for r in runs) / len(runs),
                      "outcomes": dict(Counter(r["outcome"] for r in runs))}
    return out


def task4_overfit_the_baseline() -> dict:
    peeked = re.compile(r"call sheet|stranger|address book", re.I)  # read off the test ticket: that's the problem

    def patched_first(state):
        if peeked.search(state["ticket"]):
            return {"command": "check_grain", "args": {"as_of": state["context"]["as_of"]}}
        return rules_first(state)

    patched = lambda s: next_call(s) if accepted(s) else patched_first(s)
    out = {}
    for split in ("test", "val"):
        cases = [c for c in incident_cases((NIGHT,)) if c.split == split]
        for name, policy in (("rules", rules_policy), ("patched", patched)):
            out[f"{split}_{name}"] = sum(run_loop(policy, c)["outcome"] == "solved" for c in cases) / len(cases)
    return out


def task5_ablation() -> str:
    parts = []
    for split in ("train", "test"):
        cases = all_cases(DEFAULT_NIGHTS[:2], split=split)
        parts.append(f"{split}: {len(cases)} cases\n{table(ablation(cases, default_entries()))}")
    return "\n\n".join(parts)


def main() -> None:
    print("Task 1 — three hallucinations a string schema lets through")
    for message in task1_break_the_spec():
        print("  ", message)

    print("\nTask 2 — one six-step investigation")
    print(task2_walk_one_investigation())

    print("\nTask 3 — a rejection costs a step")
    print(task3_what_a_rejection_costs())
    print("  The budget stops a policy that keeps getting rejected from running forever. A model with a 10%\n"
          "  per-call rejection rate needs ~0.6 extra steps on a six-step ticket — the budget must have that slack.")

    print("\nTask 4 — overfitting the test split")
    print(task4_overfit_the_baseline())
    print("  The keywords came from the test tickets, so the test score now measures what I read, not what the\n"
          "  baseline can do. Val was already solved, so it can't show the damage either. A model trained only on\n"
          "  train wording and scored on test wording is the comparison that still means something.")

    print("\nTask 5 — harness vs weights")
    print(task5_ablation())
    print("  Verdict: the harness is worth 0.54 → 0.99 on familiar wording (one-shot is 0 at h3, h5, h6).\n"
          "  The gap a model must close is 0.99 → 0.23: reading tickets it has never seen.")

    print("\nTask 6 — fine-tune (not in this file)")
    print("  python capstone/finetune/prepare_data.py --dry-run   # laptop")
    print("  Colab: prepare_data.py → train_lora.py → evaluate_adapter.py --adapter ... --base ...")


if __name__ == "__main__":
    main()
