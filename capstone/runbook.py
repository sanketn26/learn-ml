"""The Week 17 runbook as code: read the results so far, pick the next command.

Grain, then a column summary against last week, then one customer or a
slice, then conclude. It never sees the case's answer — only the results
the commands returned — so the trajectories it produces are what a careful
on-call engineer would do with the same screens in front of them.

The teacher (policies.teacher_policy) uses it after a privileged first
step; the rules baseline (policies.rules_policy) uses it after a keyword
guess. A model has to learn both halves: the first step from the ticket's
wording, and every later step from the numbers.
"""

from __future__ import annotations

from capstone.spec import FEATURE_COLUMNS

# Normal week-over-week movement on this data: means within ~1.5%, zero share within ~0.003.
RATIO_SHIFT = 1.5     # a mean this far from 1 is a join or a unit, not a week
UNIT_RATIO = 20.0     # ...and this far is a unit change (×100 cents, ×1000 ms)
ZERO_JUMP = 0.01      # a zero share this much higher is an extract that stopped landing for someone

CAUSE = {"unit": "unit_change", "fanout": "join_fanout", "zero_jump": "stale_extract"}
CHECK = {"unit": "mean_ratio_vs_last_week", "fanout": "mean_ratio_vs_last_week", "zero_jump": "zero_share_by_cohort"}


def _call(command: str, **args) -> dict:
    return {"command": command, "args": args}


def accepted(state: dict, command: str | None = None) -> list[dict]:
    return [s for s in state["steps"] if "result" in s and (command is None or s["command"] == command)]


def classify(shifts: list[dict]) -> dict[str, str]:
    """{column: unit | zero_jump | fanout} for every column that moved more than a normal week."""
    out = {}
    for s in shifts:
        ratio, jump = s["mean_ratio"], s["zero_share"] - s["ref_zero_share"]
        if ratio is not None and (ratio >= UNIT_RATIO or ratio <= 1 / UNIT_RATIO):
            out[s["column"]] = "unit"
        elif jump >= ZERO_JUMP:
            out[s["column"]] = "zero_jump"
        elif ratio is not None and (ratio >= RATIO_SHIFT or ratio <= 1 / RATIO_SHIFT):
            out[s["column"]] = "fanout"
    return out


def next_call(state: dict) -> dict | None:
    """The next command, or None before the first step (the runbook doesn't read tickets)."""
    done = accepted(state)
    if not done:
        return None
    ctx, last = state["context"], done[-1]
    res, n = last["result"], last["n"]

    if last["command"] == "explain_rejection":
        return _call("conclude", causes=["contract_violation"], columns=res["fields"], evidence=[n],
                     checks_to_add=["contract_test_on_caller"])
    if last["command"] == "check_leakage":
        if res["forbidden"]:
            return _call("conclude", causes=["leakage"], columns=[res["column"]], evidence=[n],
                         checks_to_add=["assert_no_forbidden"])
        return _call("escalate", reason="no_defect_found")
    if last["command"] == "check_threshold":
        if res["flagged_at_threshold"] > res["capacity"]:
            return _call("conclude", causes=["threshold_ties"], columns=[], evidence=[n],
                         checks_to_add=["tie_break_by_user_id"])
        return _call("escalate", reason="no_defect_found")

    # An incident: grain, then compare, then one customer and/or slices, then conclude.
    window = {"ref_as_of": ctx["ref_as_of"], "as_of": ctx["as_of"]}
    compared = accepted(state, "compare_nights")
    if not compared:
        return _call("compare_nights", columns=list(FEATURE_COLUMNS), **window)
    compare = compared[-1]
    moved = classify(compare["result"]["shifts"])
    if not moved:
        return _call("escalate", reason="no_defect_found")

    inspected = accepted(state, "inspect_customer")
    if any(k in ("unit", "fanout") for k in moved.values()) and not inspected:
        return _call("inspect_customer", user_id=compare["result"]["example_user_id"], **window)
    sliced = accepted(state, "slice_column")
    for column, kind in moved.items():
        if kind == "zero_jump" and column not in {s["args"]["column"] for s in sliced}:
            return _call("slice_column", column=column, by="signup_cohort", **window)

    columns = [c for c in FEATURE_COLUMNS if c in moved]
    return _call(
        "conclude",
        causes=sorted({CAUSE[moved[c]] for c in columns}),
        columns=columns,
        evidence=[compare["n"]] + [s["n"] for s in inspected + sliced],
        checks_to_add=sorted({CHECK[moved[c]] for c in columns}),
    )
