"""The command specs — one YAML file per command in capstone/commands/.

A spec is the contract for one command: what it is for, when to use it and
when not to, its typed arguments, what it returns, and whether it ends the
investigation. The validator (reliability.py), the prompt (prompt.py), the
function-calling schema, and the course page all read these files, so a
command is defined in exactly one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from pipelines.features import FEATURE_COLS, FORBIDDEN, NUMERIC

SPEC_DIR = Path(__file__).resolve().parent / "commands"

# Argument types a spec may use. Anything else is a typo in the spec, caught at load.
ARG_TYPES = {
    "date", "int", "enum", "feature_column", "column", "quote", "user_id", "steps",
    "list[enum]", "list[feature_column]", "list[column]",
}

# Numeric model inputs: the columns a distribution check can compare.
FEATURE_COLUMNS = list(NUMERIC)
# Every column a CloudWave engineer might name: inputs, forbidden keys and labels, raw extras.
KNOWN_COLUMNS = list(dict.fromkeys(FEATURE_COLS + FORBIDDEN + ["total_usage", "signup_date"]))


@lru_cache(maxsize=1)
def load_specs() -> dict[str, dict]:
    specs = {}
    for path in sorted(SPEC_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        if spec["command"] != path.stem:
            raise ValueError(f"{path.name}: command {spec['command']!r} does not match the file name")
        for arg, rule in spec["args"].items():
            if rule["type"] not in ARG_TYPES:
                raise ValueError(f"{path.name}: arg {arg!r} has unknown type {rule['type']!r}")
        specs[spec["command"]] = spec
    return specs


def terminal(command: str) -> bool:
    return bool(load_specs()[command]["terminal"])


def _signature(spec: dict) -> str:
    parts = []
    for arg, rule in spec["args"].items():
        kind = rule["type"]
        if "values" in rule:
            kind = f"{kind} of {'|'.join(rule['values'])}"
        parts.append(f"{arg}: {kind}")
    return f"{spec['command']}({', '.join(parts)})"


def catalog() -> str:
    """The command list a model sees in its prompt — signature, purpose, and the when / not-when boundary."""
    lines = [
        f"feature_column is one of {'|'.join(FEATURE_COLUMNS)}. column is one of {'|'.join(KNOWN_COLUMNS)}.",
        "Dates are YYYY-MM-DD. quote is copied verbatim from the ticket. user_id comes from an earlier result.",
        "steps are the numbers of earlier accepted steps.",
    ]
    for spec in load_specs().values():
        lines.append(f"- {_signature(spec)}: {spec['purpose']} Use when: {spec['use_when']} Not when: {spec['not_when']}")
    return "\n".join(lines)


def _json_type(rule: dict) -> dict:
    kind = rule["type"]
    if kind == "date":
        return {"type": "string", "format": "date"}
    if kind == "int":
        return {"type": "integer", "minimum": rule.get("min"), "maximum": rule.get("max")}
    if kind == "enum":
        return {"type": "string", "enum": rule["values"]}
    if kind == "feature_column":
        return {"type": "string", "enum": FEATURE_COLUMNS}
    if kind == "column":
        return {"type": "string", "enum": KNOWN_COLUMNS}
    if kind == "user_id":
        return {"type": "string", "pattern": r"^user_\d+$"}
    if kind == "steps":
        return {"type": "array", "items": {"type": "integer"}, "minItems": 1}
    if kind.startswith("list["):
        inner = _json_type({**rule, "type": kind[5:-1]})
        return {"type": "array", "items": inner, "minItems": 0 if rule.get("allow_empty") else 1}
    return {"type": "string"}


def tool_schema() -> list[dict]:
    """The same commands as a function-calling schema (llama.cpp grammar mode, or any tool-calling API)."""
    return [
        {
            "name": spec["command"],
            "description": spec["purpose"],
            "parameters": {
                "type": "object",
                "properties": {arg: _json_type(rule) for arg, rule in spec["args"].items()},
                "required": [arg for arg, rule in spec["args"].items() if rule.get("required")],
            },
        }
        for spec in load_specs().values()
    ]
