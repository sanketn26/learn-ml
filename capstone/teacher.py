"""Phase 2 — generate trajectories.

For a synthetic scenario where we already know the right tool, WE are the
teacher: correct-by-construction beats an API call. Use `generate_dataset`
to build the first few hundred examples for free. Swap in a real teacher
(a frontier model called with the same `tools.TOOLS` schema) only once you
need scale or need trajectories for cases you can't hand-write — see
docs/ml/capstone.md Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path

from capstone.reliability import validate_call
from capstone.scenarios import SCENARIOS, Scenario


def golden_call(scenario: Scenario) -> dict | None:
    """The trajectory a reliable model should have produced. None means
    "no tool call" (e.g. an injection attempt)."""
    if scenario.expect_tool == "none":
        return None
    if scenario.expect_tool == "find_potential_bugs":
        return {"name": "find_potential_bugs", "arguments": {"code": scenario.input_text, "language": "python"}}
    if scenario.expect_tool == "review_diff":
        return {"name": "review_diff", "arguments": {"diff": scenario.input_text}}
    if scenario.expect_tool == "explain_error":
        return {"name": "explain_error", "arguments": {"traceback": scenario.input_text}}
    if scenario.expect_tool == "suggest_fix":
        code, _, issue = scenario.input_text.partition("issue:")
        return {"name": "suggest_fix", "arguments": {"code": code.strip(), "issue": issue.strip()}}
    if scenario.expect_tool == "check_style_or_conventions":
        return {"name": "check_style_or_conventions", "arguments": {"code": scenario.input_text, "ruleset": "pep8"}}
    raise ValueError(f"no golden rule for tool {scenario.expect_tool!r}")


def build_trajectory(scenario: Scenario) -> dict:
    call = golden_call(scenario)
    if call is not None:
        validate_call(call)  # the teacher's own output must pass the contract
    return {"id": scenario.id, "input": scenario.input_text, "context": scenario.context, "tool_call": call}


def generate_dataset(scenarios: list[Scenario] = SCENARIOS) -> list[dict]:
    return [build_trajectory(s) for s in scenarios]


def write_splits(out_dir: Path, scenarios: list[Scenario] = SCENARIOS) -> dict[str, int]:
    """80/10/10 split by index — fine for a teaching dataset this small.
    A real run needs hundreds of scenarios per tool before this ratio means much."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset(scenarios)
    n = len(rows)
    cut_a, cut_b = int(n * 0.8), int(n * 0.9)
    splits = {"train": rows[:cut_a], "val": rows[cut_a:cut_b], "test": rows[cut_b:]}
    counts = {}
    for name, split_rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in split_rows) + ("\n" if split_rows else ""))
        counts[name] = len(split_rows)
    return counts
