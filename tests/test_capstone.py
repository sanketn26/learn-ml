"""The on-call specialist capstone: specs, validator, harness, teacher, ablation. CPU, one night."""

import json

import pytest

from capstone.cases import DEFAULT_NIGHTS, all_cases
from capstone.evaluate import ablation, default_entries, table
from capstone.harness import run_loop, run_one_shot
from capstone.policies import llm_policy, rules_planner, rules_policy, teacher_policy
from capstone.reliability import RejectedCall, validate_call
from capstone.spec import load_specs, tool_schema
from capstone.teacher import write_splits

NIGHT = DEFAULT_NIGHTS[:1]


@pytest.fixture(scope="module")
def cases():
    return all_cases(NIGHT)


def _state(ticket="score_batch died: ValueError: unknown keys: ['user_id']", steps=()):
    return {"ticket": ticket, "context": {"as_of": "2024-07-06", "ref_as_of": "2024-06-29", "capacity": 80},
            "steps": list(steps)}


def test_every_spec_loads_and_has_a_schema():
    specs = load_specs()
    assert {"conclude", "escalate"} <= set(specs)
    assert [s for s in specs.values() if s["terminal"]] and all(s["side_effects"] == "none" for s in specs.values())
    assert len(tool_schema()) == len(specs)


@pytest.mark.parametrize("call, message", [
    ({"command": "delete_repo", "args": {}}, "unknown command"),
    ({"command": "check_grain", "args": {}}, "missing required"),
    ({"command": "compare_nights", "args": {"columns": ["tenure_days"], "ref_as_of": "2024-06-29", "as_of": "2024-07-06"}},
     "unknown feature column"),
    ({"command": "compare_nights", "args": {"columns": ["mrr"], "ref_as_of": "2024-07-06", "as_of": "2024-06-29"}},
     "must be after"),
    ({"command": "inspect_customer", "args": {"user_id": "user_000417", "ref_as_of": "2024-06-29", "as_of": "2024-07-06"}},
     "does not appear"),
    ({"command": "explain_rejection", "args": {"error": "ValueError: missing keys"}}, "verbatim"),
    ({"command": "conclude", "args": {"causes": ["leakage"], "columns": [], "evidence": [1],
                                      "checks_to_add": ["assert_no_forbidden"]}}, "not accepted"),
    ({"command": "check_threshold", "args": {"capacity": "80", "as_of": "2024-07-06"}}, "integer"),
])
def test_validator_rejects_by_meaning_not_just_shape(call, message):
    with pytest.raises(RejectedCall, match=message):
        validate_call(call, _state())


def test_a_valid_call_passes_unchanged():
    call = {"command": "explain_rejection", "args": {"error": "ValueError: unknown keys: ['user_id']"}}
    assert validate_call(call, _state()) == call


def test_teacher_solves_every_case_on_real_data(cases):
    runs = [run_loop(teacher_policy(c), c) for c in cases]
    assert {r["outcome"] for r in runs} == {"solved"}
    assert {c.kind for c in cases} == {"incident", "threshold", "contract", "leakage", "unsafe", "out_of_scope"}
    assert max(r["accepted"] for r in runs) >= 6  # a three-defect night is a six-step investigation


def test_the_loop_beats_one_shot_on_long_familiar_tickets(cases):
    long = [c for c in cases if c.split == "train" and c.kind == "incident" and c.defects]
    loop = sum(run_loop(rules_policy, c)["outcome"] == "solved" for c in long)
    blind = sum(run_one_shot(rules_planner, c)["outcome"] == "solved" for c in long)
    assert loop == len(long) and blind < len(long) / 2


def test_new_wording_breaks_the_keyword_baseline(cases):
    test_incidents = [c for c in cases if c.split == "test" and c.kind == "incident"]
    assert all(run_loop(rules_policy, c)["outcome"] != "solved" for c in test_incidents)


def test_a_rejection_costs_a_step_and_a_repeat_is_rejected(cases):
    case = next(c for c in cases if c.kind == "incident" and c.defects == ("mrr_in_cents",))
    replies = iter([{"command": "check_grain", "args": {"as_of": "tonight"}}] + [case.first] * 10)
    run = run_loop(lambda state: next(replies), case, budget=4)
    steps = run["state"]["steps"]
    assert "rejected" in steps[0] and "result" in steps[1] and "repeat" in steps[2]["rejected"]
    assert run["recovered"] == 1 and run["outcome"] == "out_of_budget"


def test_llm_policy_reads_a_command_out_of_chatty_text(cases):
    case = next(c for c in cases if c.kind == "incident" and c.defects == ("events_fanout",))
    calls = iter({"command": s["command"], "args": s["args"]} for s in run_loop(teacher_policy(case), case)["state"]["steps"])
    chatty = lambda messages: f"Sure! Next I'd run:\n{json.dumps(next(calls))}\nLet me know."
    assert run_loop(llm_policy(chatty), case)["outcome"] == "solved"


def test_training_examples_pass_their_own_contract_and_split_by_wording(tmp_path):
    counts = write_splits(tmp_path, nights=NIGHT)
    assert all(counts.values())
    seen = {}
    for split in counts:
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            validate_call(row["call"], row["state"])
            assert seen.setdefault(row["case_id"], split) == split
    assert any(r.endswith("r") for r in (json.loads(l)["id"] for l in (tmp_path / "train.jsonl").read_text().splitlines()))


def test_ablation_table_renders(cases):
    few = [c for c in cases if c.split == "val"][:12]
    out = table(ablation(few, default_entries()))
    assert "teacher (ceiling)" in out and "rules, one-shot" in out
