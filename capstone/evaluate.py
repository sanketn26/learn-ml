"""Phase 5 — evaluate. Same shape as eval/router.py's golden-file check:
deterministic, no API key, a failure count you can wire into `pytest`.

Out of the box `specialist_call` is a placeholder that returns the golden
answer — it stands in for "a fine-tuned model that has fully converged" so
you can run the harness before you've trained anything. Once Phase 3 (Colab
fine-tune, see docs/ml/capstone.md) is done, point SPECIALIST_CALL at real
local inference (llama.cpp / GGUF) and rerun — the gap between this
placeholder ceiling and your real model's score is your specialization gap.
"""

from __future__ import annotations

from typing import Callable

from capstone.baseline import general_model_call
from capstone.reliability import RejectedCall, validate_call
from capstone.scenarios import SCENARIOS, Scenario
from capstone.teacher import golden_call

ModelCall = Callable[[Scenario], dict | str | None]


def specialist_call(scenario: Scenario) -> dict | None:
    return golden_call(scenario)


def score_one(scenario: Scenario, call: dict | str | None) -> dict:
    result = {"id": scenario.id, "expect_tool": scenario.expect_tool, "outcome": None}

    if scenario.expect_tool == "none":
        result["outcome"] = "correct" if call is None else "hallucinated_call"
        return result
    if call is None:
        result["outcome"] = "missing_call"
        return result
    if isinstance(call, str):
        result["outcome"] = "unstructured_output"
        return result
    if call.get("name") not in {"review_diff", "find_potential_bugs", "suggest_fix",
                                 "explain_error", "check_style_or_conventions"}:
        result["outcome"] = "hallucinated_tool"
        return result
    try:
        validate_call(call)
    except RejectedCall:
        result["outcome"] = "invalid_schema"
        return result
    result["outcome"] = "correct" if call["name"] == scenario.expect_tool else "wrong_tool"
    return result


def run(model_call: ModelCall, scenarios: list[Scenario] = SCENARIOS) -> list[dict]:
    return [score_one(s, model_call(s)) for s in scenarios]


def summarize(results: list[dict]) -> dict:
    n = len(results)
    correct = sum(r["outcome"] == "correct" for r in results)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "by_outcome": {
            outcome: sum(r["outcome"] == outcome for r in results)
            for outcome in sorted({r["outcome"] for r in results})
        },
    }


def compare() -> tuple[dict, dict]:
    specialist = summarize(run(specialist_call))
    baseline = summarize(run(lambda s: general_model_call(s, seed=0)))
    return specialist, baseline


if __name__ == "__main__":
    specialist, baseline = compare()
    print("specialist (placeholder ceiling):", specialist)
    print("baseline   (unspecialized model):", baseline)
    print()
    print(f"specialization gain (placeholder): "
          f"{specialist['accuracy'] - baseline['accuracy']:+.0%} accuracy")
