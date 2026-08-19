"""Schema validation for a proposed tool call. Same rule as Week 15/16's
`validate()`: the model is not the firewall. A call that doesn't match the
contract in tools.py never reaches a linter, a patch, or a user — it is
rejected here, deterministically, before anything downstream trusts it.
"""

from __future__ import annotations

from capstone.tools import TOOLS


class RejectedCall(ValueError):
    """A tool call that failed schema validation. Reject, don't repair —
    a repaired hallucination is still a hallucination."""


def validate_call(call: dict) -> dict:
    """call = {"name": ..., "arguments": {...}}. Returns call unchanged if
    valid; raises RejectedCall otherwise."""
    name = call.get("name")
    args = call.get("arguments")

    if name is None:
        raise RejectedCall("missing 'name'")
    if name not in TOOLS:
        raise RejectedCall(f"unknown tool {name!r} — not in the tool surface")
    if not isinstance(args, dict):
        raise RejectedCall(f"{name}: 'arguments' must be an object, got {type(args)}")

    spec = TOOLS[name]
    missing = [k for k in spec["required"] if k not in args]
    if missing:
        raise RejectedCall(f"{name}: missing required args {missing}")
    extra = [k for k in args if k not in spec["parameters"]]
    if extra:
        raise RejectedCall(f"{name}: unknown args {extra}")
    for key in args:
        if not isinstance(args[key], str):
            raise RejectedCall(f"{name}: arg {key!r} should be a string, got {type(args[key])}")

    return call
