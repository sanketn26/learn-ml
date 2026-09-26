"""Phase 5 — the ablation: is the gain in the weights or in the harness?

Every policy runs the same cases twice: `loop` (sees each result, gets its
rejections back) and `one_shot` (plans from the ticket alone). Results are
broken down by horizon — how many steps the teacher needed — because a
policy that solves two-step tickets and fails six-step ones has not learned
to investigate.

    python -m capstone.evaluate                      # teacher, rules: loop and one-shot
    python -m capstone.evaluate --ollama qwen2.5:3b  # add a local model, both modes
    python -m capstone.evaluate --nights 2 --split val

Offline, out of the box, the rows are the teacher (a ceiling, not a model)
and the rules baseline. Your model rows come from --ollama or
finetune/evaluate_adapter.py — there is no placeholder number.
"""

from __future__ import annotations

import argparse
from collections import Counter

from capstone.cases import DEFAULT_NIGHTS, Case, all_cases
from capstone.harness import run_loop, run_one_shot
from capstone.policies import llm_planner, llm_policy, ollama_generate, rules_planner, rules_policy, teacher_policy


def horizons(cases: list[Case]) -> dict[str, int]:
    """Steps the teacher needs per case — the task's length, independent of who is being graded."""
    return {c.id: run_loop(teacher_policy(c), c)["accepted"] for c in cases}


def run_suite(cases: list[Case], policy=None, planner=None, case_policy=None) -> list[dict]:
    """Exactly one of: `policy` (loop), `planner` (one-shot), `case_policy` (a policy built per case — the teacher)."""
    if case_policy is not None:
        return [run_loop(case_policy(c), c) for c in cases]
    if policy is not None:
        return [run_loop(policy, c) for c in cases]
    return [run_one_shot(planner, c) for c in cases]


def summarize(runs: list[dict], horizon: dict[str, int]) -> dict:
    n = len(runs)
    attempts = sum(r["attempts"] for r in runs)
    rejections = sum(r["rejections"] for r in runs)
    by_h: dict[int, list[bool]] = {}
    for r in runs:
        by_h.setdefault(horizon[r["id"]], []).append(r["outcome"] == "solved")
    return {
        "n": n,
        "solved": round(sum(r["outcome"] == "solved" for r in runs) / n, 3) if n else 0.0,
        "by_outcome": dict(Counter(r["outcome"] for r in runs)),
        "rejection_rate": round(rejections / attempts, 3) if attempts else 0.0,
        "recovery_rate": round(sum(r["recovered"] for r in runs) / rejections, 3) if rejections else None,
        "by_horizon": {h: round(sum(v) / len(v), 3) for h, v in sorted(by_h.items())},
    }


def ablation(cases: list[Case], entries: dict[str, dict]) -> dict[str, dict]:
    """entries: {name: {"policy" | "planner" | "case_policy": ...}} → {name: summary}."""
    horizon = horizons(cases)
    return {name: summarize(run_suite(cases, **entry), horizon) for name, entry in entries.items()}


def table(summaries: dict[str, dict]) -> str:
    hs = sorted({h for s in summaries.values() for h in s["by_horizon"]})
    head = f"{'policy':<24}{'solved':>8}{'reject':>8}{'recover':>9}  " + "".join(f"{'h' + str(h):>6}" for h in hs)
    lines = [head, "-" * len(head)]
    for name, s in summaries.items():
        recover = "—" if s["recovery_rate"] is None else f"{s['recovery_rate']:.2f}"
        cells = "".join(f"{s['by_horizon'].get(h, float('nan')):>6.2f}" for h in hs)
        lines.append(f"{name:<24}{s['solved']:>8.2f}{s['rejection_rate']:>8.2f}{recover:>9}  {cells}")
    return "\n".join(lines)


def default_entries() -> dict[str, dict]:
    return {
        "teacher (ceiling)": {"case_policy": teacher_policy},
        "rules + loop": {"policy": rules_policy},
        "rules, one-shot": {"planner": rules_planner},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness vs weights, by horizon.")
    parser.add_argument("--split", default="train,test", help="comma-separated: train (familiar wording), val, test (new wording)")
    parser.add_argument("--nights", type=int, default=2, help="use the first N incident nights (max %d)" % len(DEFAULT_NIGHTS))
    parser.add_argument("--ollama", action="append", default=[], help="a local model name; repeatable")
    args = parser.parse_args()

    entries = default_entries()
    for model in args.ollama:
        generate = ollama_generate(model)
        entries[f"{model} + loop"] = {"policy": llm_policy(generate)}
        entries[f"{model}, one-shot"] = {"planner": llm_planner(generate)}

    for split in args.split.split(","):
        cases = all_cases(DEFAULT_NIGHTS[: args.nights], split=split)
        print(f"\n{split}: {len(cases)} cases {dict(Counter(c.kind for c in cases))}\n")
        print(table(ablation(cases, entries)))


if __name__ == "__main__":
    main()
