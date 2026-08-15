"""Deterministic tool allowlist. The LLM is not the firewall."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN = Path(__file__).with_name("golden_tickets.jsonl")


def allowed_tools(question: str) -> list[str]:
    q = question.lower()
    if any(p in q for p in ("ignore previous", "refund", "export all", "email addresses")):
        return []
    if any(p in q for p in ("cancel", "churn", "about to leave")):
        return ["get_churn_score"]
    return []


def evaluate(path: Path = GOLDEN) -> int:
    failures = 0
    for line in path.read_text().splitlines():
        case = json.loads(line)
        tools = allowed_tools(case["question"])
        if case["expect_tool"] == "none" and tools:
            print("FAIL", case["id"], "expected no tools, got", tools)
            failures += 1
        elif case["expect_tool"] != "none" and case["expect_tool"] not in tools:
            print("FAIL", case["id"], "missing", case["expect_tool"], "got", tools)
            failures += 1
        if any(bad in tools for bad in case.get("must_not", [])):
            print("FAIL", case["id"], "forbidden tool in", tools)
            failures += 1
    return failures


if __name__ == "__main__":
    n = evaluate()
    print("failures", n)
    raise SystemExit(n)
