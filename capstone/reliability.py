"""The firewall between the model and anything that runs. Same rule as Week
15's `validate()`: the model is not the firewall. A call that doesn't match
its spec in capstone/commands/ never executes — it is rejected here,
deterministically, and the rejection goes back into the model's state.

Shape is not enough. `validate_call` also checks meaning against the
investigation so far: a quote must really be in the ticket, a user_id must
have come from an earlier result, and evidence must point at steps that
were accepted. Those are the hallucinations a string-typed schema lets through.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from capstone.spec import FEATURE_COLUMNS, KNOWN_COLUMNS, load_specs
from pipelines.features import OBSERVATION_END

DATE_MIN = pd.Timestamp("2023-01-01")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
USER = re.compile(r"^user_\d+$")


class RejectedCall(ValueError):
    """A command call that failed its spec. Reject, don't repair —
    a repaired hallucination is still a hallucination."""


def _accepted(state: dict) -> list[dict]:
    return [s for s in state["steps"] if "result" in s]


def _check(command: str, arg: str, value, rule: dict, args: dict, state: dict | None) -> None:
    kind = rule["type"]
    where = f"{command}.{arg}"

    if kind.startswith("list["):
        if not isinstance(value, list):
            raise RejectedCall(f"{where}: expected a list, got {type(value).__name__}")
        if not value and not rule.get("allow_empty"):
            raise RejectedCall(f"{where}: must not be empty")
        if len(set(map(str, value))) != len(value):
            raise RejectedCall(f"{where}: duplicate entries")
        for item in value:
            _check(command, arg, item, {**rule, "type": kind[5:-1]}, args, state)
        return

    if kind == "date":
        if not isinstance(value, str) or not DATE.match(value):
            raise RejectedCall(f"{where}: {value!r} is not a YYYY-MM-DD date")
        day = pd.Timestamp(value)
        if not DATE_MIN <= day <= OBSERVATION_END:
            raise RejectedCall(f"{where}: {value} is outside the data ({DATE_MIN.date()} to {OBSERVATION_END.date()})")
        if "after" in rule and isinstance(args.get(rule["after"]), str) and day <= pd.Timestamp(args[rule["after"]]):
            raise RejectedCall(f"{where}: {value} must be after {rule['after']} ({args[rule['after']]})")
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise RejectedCall(f"{where}: expected an integer, got {value!r}")
        if not rule.get("min", value) <= value <= rule.get("max", value):
            raise RejectedCall(f"{where}: {value} outside [{rule.get('min')}, {rule.get('max')}]")
    elif kind == "enum":
        if value not in rule["values"]:
            raise RejectedCall(f"{where}: {value!r} is not one of {rule['values']}")
    elif kind == "feature_column":
        if value not in FEATURE_COLUMNS:
            raise RejectedCall(f"{where}: unknown feature column {value!r}")
    elif kind == "column":
        if value not in KNOWN_COLUMNS:
            raise RejectedCall(f"{where}: unknown column {value!r}")
    elif kind == "quote":
        if not isinstance(value, str) or len(value.strip()) < 8:
            raise RejectedCall(f"{where}: quote the error text, at least 8 characters")
        if state is not None and value not in state["ticket"]:
            raise RejectedCall(f"{where}: not a verbatim quote from the ticket")
    elif kind == "user_id":
        if not isinstance(value, str) or not USER.match(value):
            raise RejectedCall(f"{where}: {value!r} is not a user_id")
        seen = state is not None and any(value in json.dumps(s["result"]) for s in _accepted(state))
        if state is not None and not seen:
            raise RejectedCall(f"{where}: {value} does not appear in any earlier result")
    elif kind == "steps":
        if not isinstance(value, list) or not value or not all(isinstance(n, int) and not isinstance(n, bool) for n in value):
            raise RejectedCall(f"{where}: expected a non-empty list of step numbers")
        if state is not None:
            ok = {s["n"] for s in _accepted(state)}
            bad = [n for n in value if n not in ok]
            if bad:
                raise RejectedCall(f"{where}: steps {bad} were not accepted steps")


def validate_call(call, state: dict | None = None) -> dict:
    """call = {"command": ..., "args": {...}}. Returns it unchanged if valid; raises RejectedCall otherwise.

    With `state`, also checks the arguments against the investigation so far. The harness always passes it.
    """
    if not isinstance(call, dict):
        raise RejectedCall(f"expected a JSON object, got {type(call).__name__}")
    extra = set(call) - {"command", "args"}
    if extra:
        raise RejectedCall(f"unknown top-level keys {sorted(extra)}")
    command, args = call.get("command"), call.get("args")
    if command is None:
        raise RejectedCall("missing 'command'")
    specs = load_specs()
    if command not in specs:
        raise RejectedCall(f"unknown command {command!r} — not in the command surface")
    if not isinstance(args, dict):
        raise RejectedCall(f"{command}: 'args' must be an object")

    rules = specs[command]["args"]
    missing = [a for a, r in rules.items() if r.get("required") and a not in args]
    if missing:
        raise RejectedCall(f"{command}: missing required args {missing}")
    unknown = [a for a in args if a not in rules]
    if unknown:
        raise RejectedCall(f"{command}: unknown args {unknown}")
    for arg, value in args.items():
        _check(command, arg, value, rules[arg], args, state)
    return call
