"""Who picks the next command.

- `teacher_policy(case)` — knows the right first command (privileged), then
  follows the runbook on real results. It generates the training data and is
  the ceiling. It is not a model.
- `rules_policy` / `rules_planner` — a keyword guess at the first command,
  then the same runbook. The honest non-ML baseline: it reads numbers
  perfectly and tickets badly.
- `llm_policy(generate)` / `llm_planner(generate)` — any chat model behind a
  `generate(messages) -> str` function: a general model, your fine-tuned
  specialist, or a frontier model. Same prompt format as the training data.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable

from capstone.cases import Case
from capstone.execute import explain_rejection
from capstone.prompt import messages_for, parse_call, parse_plan, plan_messages
from capstone.runbook import accepted, next_call
from capstone.spec import FEATURE_COLUMNS, KNOWN_COLUMNS

Generate = Callable[[list[dict]], str]


def teacher_policy(case: Case):
    def policy(state: dict) -> dict:
        return next_call(state) if accepted(state) else case.first
    return policy


UNSAFE = re.compile(r"ignore previous|delete|drop the|disable|turn off|skip|overwrite|without the backtest", re.I)
ERROR = re.compile(r"((?:ValueError|TypeError): .+)")
DIFF_COLUMN = re.compile(r'^\+\s+"(\w+)"', re.M)
ASK_COLUMN = re.compile(r"(?:add|train on) (\w+)", re.I)
THRESHOLD = re.compile(r"flag|threshold|different", re.I)
INCIDENT = re.compile(r"\blist\b|scores?\b|names", re.I)


def _call(command: str, **args) -> dict:
    return {"command": command, "args": args}


def rules_first(state: dict) -> dict:
    """A keyword guess at the first command — written from the training tickets, as an engineer would."""
    text, ctx = state["ticket"], state["context"]
    if UNSAFE.search(text):
        return _call("escalate", reason="unsafe_request")
    if match := ERROR.search(text):
        return _call("explain_rejection", error=match.group(1).strip())
    for pattern in (DIFF_COLUMN, ASK_COLUMN):
        match = pattern.search(text)
        if match and match.group(1) in KNOWN_COLUMNS:
            return _call("check_leakage", column=match.group(1))
    if THRESHOLD.search(text):
        return _call("check_threshold", capacity=ctx["capacity"], as_of=ctx["as_of"])
    if INCIDENT.search(text):
        return _call("check_grain", as_of=ctx["as_of"])
    return _call("escalate", reason="out_of_scope")


def rules_policy(state: dict) -> dict:
    return next_call(state) if accepted(state) else rules_first(state)


def rules_planner(state: dict) -> list[dict]:
    """The rules baseline without the loop: it must guess every later step without seeing a result."""
    first, ctx = rules_first(state), state["context"]
    command = first["command"]
    if command == "escalate":
        return [first]
    if command == "explain_rejection":
        # validate()'s message format is public, so the field can be read off the ticket without running anything.
        fields = explain_rejection(None, first["args"]["error"])["fields"]
        return [first, _call("conclude", causes=["contract_violation"], columns=fields, evidence=[1],
                             checks_to_add=["contract_test_on_caller"])]
    if command == "check_leakage":
        return [first, _call("conclude", causes=["leakage"], columns=[first["args"]["column"]], evidence=[1],
                             checks_to_add=["assert_no_forbidden"])]
    if command == "check_threshold":
        return [first, _call("conclude", causes=["threshold_ties"], columns=[], evidence=[1],
                             checks_to_add=["tie_break_by_user_id"])]
    # An incident, blind: the most common cause in the training bank is a fan-out.
    window = {"ref_as_of": ctx["ref_as_of"], "as_of": ctx["as_of"]}
    return [first, _call("compare_nights", columns=list(FEATURE_COLUMNS), **window),
            _call("conclude", causes=["join_fanout"], columns=["total_events", "n_support"], evidence=[1, 2],
                  checks_to_add=["mean_ratio_vs_last_week"])]


def llm_policy(generate: Generate):
    def policy(state: dict):
        return parse_call(generate(messages_for(state)))
    return policy


def llm_planner(generate: Generate):
    def planner(state: dict):
        return parse_plan(generate(plan_messages(state)))
    return planner


def ollama_generate(model: str, host: str = "http://localhost:11434", timeout: float = 120.0) -> Generate:
    """A local model through Ollama's chat API — a GGUF from Phase 3, or any general model you pulled."""
    def generate(messages: list[dict]) -> str:
        body = json.dumps({"model": model, "messages": messages, "stream": False,
                           "options": {"temperature": 0}}).encode()
        request = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["message"]["content"]
    return generate
