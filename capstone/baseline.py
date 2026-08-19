"""A stand-in for "prompt a general model, no fine-tuning, no schema
enforcement." Seeded and deterministic so the evaluation harness (Phase 5)
is runnable offline before you have any real model wired up — swap
`general_model_call` for a real API call once you're comparing for real.

The failure modes below (wrong tool, hallucinated tool, free text instead of
JSON, missing arg) are the ones the write-up in docs/ml/capstone.md asks you
to quantify — this module exists to give you a baseline to quantify *against*.
"""

from __future__ import annotations

import random

from capstone.scenarios import Scenario
from capstone.teacher import golden_call

_FAILURE_MODES = ["correct", "wrong_tool", "hallucinated_tool", "free_text", "missing_arg"]
_WEIGHTS = [0.45, 0.2, 0.1, 0.15, 0.1]  # a plausible, not-great general-model profile


def general_model_call(scenario: Scenario, seed: int = 0) -> dict | str | None:
    """Simulates an unspecialized model's response to `scenario`."""
    rng = random.Random(f"{scenario.id}-{seed}")
    mode = rng.choices(_FAILURE_MODES, weights=_WEIGHTS, k=1)[0]
    golden = golden_call(scenario)

    if scenario.expect_tool == "none":
        # injection resistance is exactly where general models are weakest
        return None if mode in ("correct", "missing_arg") else golden_call_for_wrong_tool(rng)
    if mode == "correct":
        return golden
    if mode == "wrong_tool":
        return golden_call_for_wrong_tool(rng, exclude=scenario.expect_tool)
    if mode == "hallucinated_tool":
        return {"name": "run_shell_command", "arguments": {"cmd": "rm -rf ."}}
    if mode == "free_text":
        return f"I looked at this and I think the issue is probably fine, here's my analysis: {scenario.context}"
    if mode == "missing_arg" and golden is not None:
        arguments = dict(golden["arguments"])
        arguments.pop(next(iter(arguments)))
        return {"name": golden["name"], "arguments": arguments}
    return golden


def golden_call_for_wrong_tool(rng: random.Random, exclude: str | None = None) -> dict:
    from capstone.tools import TOOLS

    choices = [t for t in TOOLS if t != exclude]
    name = rng.choice(choices)
    spec = TOOLS[name]
    arguments = {k: "placeholder" for k in spec["required"]}
    return {"name": name, "arguments": arguments}
