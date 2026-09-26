"""The loop that makes a long investigation a chain of short decisions.

`run_loop` asks a policy for one command at a time. Each call is validated
against its spec, checked for repeats, executed, and its result (or its
rejection) appended to the state the policy sees next. It stops on a
terminal command or when the step budget runs out. A rejection costs a
step — the model gets to fix its call, but not for free.

`run_one_shot` is the ablation: the planner writes every step up front from
the ticket alone and never sees a result. Same commands, same validator,
no feedback. The gap between the two is what the harness is worth.
"""

from __future__ import annotations

from typing import Callable

from capstone.cases import Case
from capstone.execute import CommandError, Night, execute
from capstone.reliability import RejectedCall, validate_call
from capstone.runbook import accepted
from capstone.spec import terminal

BUDGET = 8

Policy = Callable[[dict], object]          # state -> one call (or raw text)
Planner = Callable[[dict], object]         # state -> list of calls (or raw text)


def new_state(case: Case) -> dict:
    return {"ticket": case.ticket, "context": dict(case.context), "steps": []}


def _repeat(call: dict, state: dict) -> bool:
    return any(s["command"] == call["command"] and s["args"] == call["args"] for s in accepted(state))


def attempt(call, state: dict, night: Night) -> bool:
    """Validate, execute, and record one call. Returns True if it was accepted."""
    n = len(state["steps"]) + 1
    try:
        validate_call(call, state)
        if _repeat(call, state):
            raise RejectedCall("repeat: same command and args as an earlier step")
        result = execute(call, night)
    except (RejectedCall, CommandError) as exc:
        state["steps"].append({"n": n, "call": call, "rejected": str(exc)})
        return False
    state["steps"].append({"n": n, "command": call["command"], "args": call["args"], "result": result})
    return True


def grade(case: Case, state: dict) -> str:
    done = accepted(state)
    final = done[-1] if done and terminal(done[-1]["command"]) else None
    if final is None:
        return "out_of_budget"
    if "escalate" in case.expect:
        if final["command"] != "escalate":
            return "should_have_escalated"
        # "no defect found" is only true after you looked; "unsafe" is not the same hand-off as "out of scope".
        return "solved" if final["args"]["reason"] == case.expect["escalate"] else "wrong_escalation_reason"
    if final["command"] == "escalate":
        return "escalated_wrongly"
    args = final["args"]
    right = set(args["causes"]) == set(case.expect["causes"]) and set(args["columns"]) == set(case.expect["columns"])
    return "solved" if right else "wrong_diagnosis"


def _record(case: Case, state: dict, mode: str) -> dict:
    steps = state["steps"]
    rejected = [i for i, s in enumerate(steps) if "rejected" in s]
    recovered = sum(1 for i in rejected if i + 1 < len(steps) and "result" in steps[i + 1])
    return {
        "id": case.id, "kind": case.kind, "split": case.split, "mode": mode,
        "outcome": grade(case, state), "attempts": len(steps), "accepted": len(steps) - len(rejected),
        "rejections": len(rejected), "recovered": recovered, "state": state,
    }


def run_loop(policy: Policy, case: Case, budget: int = BUDGET, night: Night | None = None) -> dict:
    night = night or case.night()
    state = new_state(case)
    while len(state["steps"]) < budget:
        try:
            call = policy(state)
        except Exception as exc:  # a crashed policy is a rejected step, not a crashed harness
            call = f"policy error: {exc}"
        if attempt(call, state, night) and terminal(state["steps"][-1]["command"]):
            break
    return _record(case, state, "loop")


def run_one_shot(planner: Planner, case: Case, budget: int = BUDGET, night: Night | None = None) -> dict:
    night = night or case.night()
    state = new_state(case)
    try:
        plan = planner(new_state(case))
    except Exception as exc:
        plan = f"planner error: {exc}"
    if not isinstance(plan, list):
        plan = [plan]
    for call in plan[:budget]:
        if attempt(call, state, night) and terminal(state["steps"][-1]["command"]):
            break
    return _record(case, state, "one_shot")
