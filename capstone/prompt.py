"""What a model sees, and how its reply is read back.

Training (finetune/prepare_data.py) and inference (policies.llm_policy) both
build messages here, so a fine-tuned model is evaluated on exactly the
format it was trained on.
"""

from __future__ import annotations

import json

from capstone.spec import catalog

SYSTEM = (
    "You are CloudWave's pipeline on-call specialist. You investigate one ticket by running read-only "
    "commands, one per turn. Reply with exactly one JSON object: "
    '{"command": "<name>", "args": {...}}. Use only the commands below. Read the results of earlier '
    "steps before choosing the next one. If a step was rejected, fix the call. Conclude only on evidence "
    "from accepted steps; escalate rather than guess."
)

PLAN_SYSTEM = (
    "You are CloudWave's pipeline on-call specialist. Plan the whole investigation of one ticket up front: "
    "you will not see any results. Reply with one JSON array of command objects, "
    '[{"command": "<name>", "args": {...}}, ...], ending with conclude or escalate. Steps are numbered from 1.'
)


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def render_state(state: dict) -> str:
    ctx = " ".join(f"{k}={v}" for k, v in state["context"].items())
    lines = [f"TICKET: {state['ticket']}", f"CONTEXT: {ctx}"]
    for step in state["steps"]:
        if "result" in step:
            lines.append(f"STEP {step['n']}: {_compact({'command': step['command'], 'args': step['args']})}")
            lines.append(f"RESULT {step['n']}: {_compact(step['result'])}")
        else:
            lines.append(f"STEP {step['n']} REJECTED: {_compact(step['call'])} -> {step['rejected']}")
    return "\n".join(lines)


def messages_for(state: dict) -> list[dict]:
    return [
        {"role": "system", "content": f"{SYSTEM}\n\nCOMMANDS\n{catalog()}"},
        {"role": "user", "content": f"{render_state(state)}\nNEXT COMMAND:"},
    ]


def plan_messages(state: dict) -> list[dict]:
    return [
        {"role": "system", "content": f"{PLAN_SYSTEM}\n\nCOMMANDS\n{catalog()}"},
        {"role": "user", "content": f"{render_state(state)}\nPLAN:"},
    ]


def _first_json(text: str, opener: str):
    decoder = json.JSONDecoder()
    start = text.find(opener)
    while start != -1:
        try:
            return decoder.raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            start = text.find(opener, start + 1)
    return None


def parse_call(text: str):
    """The first JSON object in a reply, or the raw text if there is none (the harness rejects it)."""
    found = _first_json(text, "{")
    return found if isinstance(found, dict) else text


def parse_plan(text: str):
    found = _first_json(text, "[")
    return found if isinstance(found, list) else text
