"""Phase 2 — turn solved investigations into training examples.

The teacher solves every case in the bank on real data. Each accepted step
becomes one example: (the state before it → the command it chose). A
model trained on these learns single decisions; the harness chains them.

Some examples also get a rejected attempt spliced in before the right
answer — a hallucinated column, swapped dates, an invented user_id — with
the validator's real message. That is how a small model learns to recover
from a rejection instead of repeating it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from capstone.cases import DEFAULT_NIGHTS, Case, all_cases
from capstone.harness import run_loop
from capstone.policies import teacher_policy
from capstone.reliability import RejectedCall, validate_call

RECOVERY_RATE = 0.2


def trajectory(case: Case) -> dict:
    run = run_loop(teacher_policy(case), case)
    if run["outcome"] != "solved":
        raise AssertionError(f"teacher failed {case.id}: {run['outcome']}")
    return run


def corrupt(call: dict) -> dict:
    """A plausible wrong version of `call` — the mistakes small models actually make."""
    bad = copy.deepcopy(call)
    args, command = bad["args"], bad["command"]
    if "ref_as_of" in args:
        args["ref_as_of"], args["as_of"] = args["as_of"], args["ref_as_of"]
    elif command == "check_grain":
        args["as_of"] = "last week"
    elif command == "check_threshold":
        args["capacity"] = str(args["capacity"])
    elif command == "check_leakage":
        args["column"] = args["column"].split("_")[0]
    elif command == "explain_rejection":
        args["error"] = "the payload was rejected"
    elif command == "conclude":
        args["evidence"] = args["evidence"] + [99]
    elif command == "escalate":
        args["reason"] = "unsure"
    return bad


def examples(case: Case, run: dict, rng: random.Random) -> list[dict]:
    steps, rows = run["state"]["steps"], []
    base = {"case_id": case.id, "kind": case.kind, "split": case.split}
    for i, step in enumerate(steps):
        before = {**run["state"], "steps": steps[:i]}
        target = {"command": step["command"], "args": step["args"]}
        rows.append({**base, "id": f"{case.id}#{step['n']}", "state": before, "call": target})
        if rng.random() < RECOVERY_RATE:
            bad = corrupt(target)
            try:
                validate_call(bad, before)
            except RejectedCall as exc:
                rejected = {"n": len(before["steps"]) + 1, "call": bad, "rejected": str(exc)}
                rows.append({**base, "id": f"{case.id}#{step['n']}r", "recovery": True,
                             "state": {**before, "steps": before["steps"] + [rejected]}, "call": target})
    return rows


def write_splits(out_dir: Path, nights=DEFAULT_NIGHTS, seed: int = 0) -> dict[str, int]:
    """Solve every case, write {train,val,test}.jsonl of per-step examples, return example counts."""
    rng = random.Random(seed)
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for case in all_cases(nights):
        rows[case.split].extend(examples(case, trajectory(case), rng))
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split_rows))
    return {split: len(split_rows) for split, split_rows in rows.items()}
