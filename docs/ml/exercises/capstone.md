---
description: Capstone exercises — break a command spec on purpose, walk a six-step investigation through the harness, measure what a rejection costs, overfit a keyword baseline to see why it fails, run the harness-versus-weights ablation, and (on a GPU) fine-tune the on-call specialist.
---

# Exercises — Capstone — The On-Call Specialist

Everything except Task 6 runs on a laptop, offline, in about a minute. Task 6 is the GPU step.

## What you are building

Three hallucinated calls the validator catches by meaning, one long investigation read step by step, a measurement of what a rejection costs, a lesson in overfitting a test split, an ablation table with a verdict, and — if you have a GPU — a fine-tuned specialist scored against its own base model.

## Predict before you run

1. `{"command": "compare_nights", "args": {"columns": ["tenure_days"], ...}}` is valid JSON with string arguments. Will `validate_call` accept it?
2. On a night with three seeded defects, how many steps does the teacher need? How many would a policy that can't see results get right?
3. The rules baseline solves 0.99 of the training tickets. What will it score on tickets worded in ways it has never seen?
4. If every decision is 95% right, what share of six-step investigations come out right?

## Before you start

- Run from the repo root in the main venv. The first `check_threshold` trains the production model once into `artifacts/capstone-commands/` (gitignored); after that everything is cached.
- `capstone/runbook.py` and `capstone/cases.py` are the answer key for which command comes next. Read the specs in `capstone/commands/` first — that's what a model sees.
- Do not `import unsloth` (or `peft`, `trl`, `bitsandbytes`) on the laptop. Those are Task 6 only, in Colab.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `exercises/ml/capstone/starter.py` and the `capstone/` package. Run from the repo root:

```bash
python exercises/ml/capstone/starter.py
python -m capstone.evaluate
pytest tests/test_capstone.py
python capstone/finetune/prepare_data.py --dry-run
```

**1. Break a spec on purpose.** Write three calls that are well-formed JSON with the right argument names, and that `validate_call(call, state)` still rejects: a column that sounds right but isn't a feature, a customer nobody showed you, and a `conclude` whose evidence cites a step that was rejected. For each, name the line in `capstone/reliability.py` that fired, and the spec field it enforces.

??? tip "Hint 1 — a nudge"
    The old five-tool contract typed every argument as `str`, so it could only reject the wrong *shape*. Which of these three would a string-typed schema have let through? Now look at the arg types in `capstone/commands/*.yaml` — which types need to know what happened earlier in the investigation?

??? tip "Hint 2 — the approach"
    Build a `state` dict (`ticket`, `context`, `steps`) with one accepted step and one rejected step in it — the harness's own format: accepted steps have `command`, `args`, `result`; rejected steps have `call` and `rejected`. Then call `validate_call(call, state)` inside `try` / `except RejectedCall as exc` and print `exc`. The `feature_column`, `user_id`, and `steps` branches of `_check` are the three you're after.

??? example "Hint 3 — most of the code"
    ```python
    import json
    from collections import Counter

    from capstone.cases import DEFAULT_NIGHTS, all_cases, incident_cases
    from capstone.evaluate import ablation, default_entries, table
    from capstone.harness import run_loop, run_one_shot
    from capstone.policies import rules_planner, rules_policy, teacher_policy
    from capstone.prompt import render_state
    from capstone.reliability import RejectedCall, validate_call
    from capstone.teacher import corrupt

    state = {
        "ticket": "Tonight's list barely overlaps last week's.",
        "context": {"as_of": "2024-07-06", "ref_as_of": "2024-06-29", "capacity": 80},
        "steps": [
            {"n": 1, "command": "check_grain", "args": {"as_of": "2024-07-06"},
             "result": {"rows": 28475, "duplicate_user_ids": 0, "grain_ok": True}},
            {"n": 2, "call": {"command": "check_grain", "args": {"as_of": "tonight"}}, "rejected": "not a date"},
        ],
    }
    window = {"ref_as_of": "2024-06-29", "as_of": "2024-07-06"}
    broken = [
        {"command": "compare_nights", "args": {"columns": ["tenure_days"], **window}},
        {"command": "inspect_customer", "args": {"user_id": "user_000417", **window}},
        {"command": "conclude", "args": {"causes": ["join_fanout"], "columns": ["total_events"], "evidence": [2],
                                         "checks_to_add": ["mean_ratio_vs_last_week"]}},
    ]
    for call in broken:
        try:
            validate_call(call, state)
            print("ACCEPTED?!", call["command"])
        except RejectedCall as exc:
            print(f"{call['command']:<17} → {exc}")
    ```
    Which spec field each message enforces — and why the second one is the hallucination a small model makes most — is yours to write.

**2. Walk one long investigation.** Take the night of `2024-07-06` with all three defects seeded. Run the teacher through `run_loop` and print the state with `render_state`. For each step, write one line: what the result showed, and why the next command follows from it. Then run `rules_policy` on the *test-split* wording of the same night and say at which step it goes wrong.

??? tip "Hint 1 — a nudge"
    Three defects touch five columns. Which of them does a mean ratio catch, and which only show up as a zero share? How many different screens does that make you open before you can conclude?

??? tip "Hint 2 — the approach"
    `incident_cases(("2024-07-06",))` builds every incident for that night; pick the one whose `defects` has length 3 and whose `split` is `"train"`, and the one with the same defects in `"test"`. `run_loop(teacher_policy(case), case)` returns a record with `outcome` and `state`. For the rules run, look at the first step only — the rest of the runbook is the same code the teacher uses.

??? example "Hint 3 — most of the code"
    ```python
    night = incident_cases(("2024-07-06",))
    three = [c for c in night if len(c.defects) == 3]
    familiar = next(c for c in three if c.split == "train")
    unseen = next(c for c in three if c.split == "test")

    run = run_loop(teacher_policy(familiar), familiar)
    print(run["outcome"], run["accepted"], "steps")
    print(render_state(run["state"]))

    rules = run_loop(rules_policy, unseen)
    print("\nnew wording:", unseen.ticket)
    print("rules:", rules["outcome"], "first step:", rules["state"]["steps"][0].get("command"))
    ```
    The one-line explanation per step is the exercise. Step 2's result is the one to slow down on.

**3. What a rejection costs.** Write a *sloppy* policy: it runs the teacher, but its first call in every investigation is wrong in a realistic way (`capstone.teacher.corrupt` makes one). Run it on the training incidents twice — with the default budget of 8, and with a budget equal to the teacher's step count. Report solved rate and recovery rate for both, and say what the budget is protecting against.

??? tip "Hint 1 — a nudge"
    The harness charges a step for a rejection. A policy that always recovers still spends one extra step per mistake. What happens when the budget has no slack?

??? tip "Hint 2 — the approach"
    Wrap `teacher_policy(case)`: if the state has no steps yet, return `corrupt(teacher(state))`; otherwise return `teacher(state)`. The rejection lands in the state, so on the next turn the teacher sees one rejected step and zero accepted ones and makes its real first call. `run_loop(policy, case, budget=...)` takes the budget; each record has `outcome`, `rejections`, and `recovered`.

??? example "Hint 3 — most of the code"
    ```python
    def sloppy(case):
        teacher = teacher_policy(case)
        def policy(state):
            call = teacher(state)
            return corrupt(call) if not state["steps"] else call
        return policy

    train_incidents = [c for c in night if c.split == "train" and c.defects]
    horizon = {c.id: run_loop(teacher_policy(c), c)["accepted"] for c in train_incidents}
    for label, budget in (("budget 8", lambda c: 8), ("no slack", lambda c: horizon[c.id])):
        runs = [run_loop(sloppy(c), c, budget=budget(c)) for c in train_incidents]
        solved = sum(r["outcome"] == "solved" for r in runs) / len(runs)
        recovered = sum(r["recovered"] for r in runs) / max(sum(r["rejections"] for r in runs), 1)
        print(f"{label:<9} solved={solved:.2f} recovery={recovered:.2f} outcomes={Counter(r['outcome'] for r in runs)}")
    ```
    The sentence about what the budget protects against — and what it would cost a real model with a 10% rejection rate at six steps — is yours.

**4. Overfit the keyword baseline — then explain why you shouldn't.** The rules baseline collapses on test-split wording. Make it better: add keywords to `rules_first` (in a copy, not in `capstone/policies.py`) until it solves most of the test incidents. Then write three sentences on what you just did to the test split, and why a fine-tuned model evaluated on held-out wording is a fairer comparison than your patched rules.

??? tip "Hint 1 — a nudge"
    Where did you get the new keywords from? If the answer is "from reading the test tickets", what is your test score now measuring?

??? tip "Hint 2 — the approach"
    Write `patched_first(state)` that checks for your new words first and otherwise defers to `capstone.policies.rules_first`. A loop policy is then `lambda s: next_call(s) if accepted(s) else patched_first(s)` — `next_call` and `accepted` live in `capstone.runbook`. Compare solved rates on the test incidents before and after, then do the same on the *val* split, which your keywords were not written from.

??? example "Hint 3 — most of the code"
    ```python
    import re

    from capstone.policies import rules_first
    from capstone.runbook import accepted, next_call

    PEEKED = re.compile(r"call sheet|stranger|address book", re.I)  # read straight off the test ticket

    def patched_first(state):
        if PEEKED.search(state["ticket"]):
            return {"command": "check_grain", "args": {"as_of": state["context"]["as_of"]}}
        return rules_first(state)

    patched = lambda s: next_call(s) if accepted(s) else patched_first(s)
    for split in ("test", "val"):
        incidents = [c for c in night if c.split == split]
        for name, policy in (("rules", rules_policy), ("patched", patched)):
            solved = sum(run_loop(policy, c)["outcome"] == "solved" for c in incidents) / len(incidents)
            print(f"{split:<5} {name:<8} solved={solved:.2f}")
    ```
    The three sentences are the point. A number that went up because you read the answer key is not a result.

**5. Run the ablation and write the verdict.** Run `python -m capstone.evaluate`. From the two tables, write down (a) how much the harness is worth on its own, with the horizon where one-shot falls to zero, and (b) which gap is left for a model to close. If you have [Ollama](https://ollama.com), add a general model with `--ollama <name>` and fill its two rows.

??? tip "Hint 1 — a nudge"
    Compare rows that differ in exactly one thing. `rules + loop` and `rules, one-shot` differ only in whether the policy sees results. `rules + loop` on train and on test differ only in the wording of the tickets.

??? tip "Hint 2 — the approach"
    `ablation(cases, entries)` returns one summary per row with `solved`, `rejection_rate`, `recovery_rate`, and `by_horizon`; `table(...)` prints them. Run it per split. The harness's worth is loop minus one-shot on the train split; the gap left for weights is train minus test for `rules + loop`. With Ollama, `capstone.policies.llm_policy(ollama_generate(name))` and `llm_planner(...)` are the two rows.

??? example "Hint 3 — most of the code"
    ```python
    for split in ("train", "test"):
        cases = all_cases(DEFAULT_NIGHTS[:1], split=split)
        summaries = ablation(cases, default_entries())
        print(f"\n{split}: {len(cases)} cases")
        print(table(summaries))

    # with Ollama running:
    # from capstone.policies import llm_planner, llm_policy, ollama_generate
    # gen = ollama_generate("qwen2.5:3b")
    # entries = {**default_entries(), "qwen + loop": {"policy": llm_policy(gen)}, "qwen, one-shot": {"planner": llm_planner(gen)}}
    ```
    Write the verdict as two numbers and one sentence: *the harness is worth X; the model has to close Y; a general model closes Z of it.*

**6. Fine-tune, and prove where the gain came from (GPU).** Generate the full bank, format it, train a LoRA adapter on Colab, and score it with `evaluate_adapter.py --adapter ... --base ...`. Fill in the four-row table from the lesson's Phase 5 on the test split, then write a four-sentence ship memo that names the base model you'd keep and the horizon column that decided it.

??? tip "Hint 1 — a nudge"
    The row that matters most isn't your adapter — it's the base model *with* the loop. Without it, you can't say whether fine-tuning did anything the harness wasn't already doing.

??? tip "Hint 2 — the approach"
    On a laptop: `prepare_data.py --dry-run` and `train_lora.py --dry-run` prove the pipeline. On Colab: clone, `pip install -r requirements-capstone.txt`, `prepare_data.py` (the full bank, ~40 s), `train_lora.py --model unsloth/functiongemma-270m-it`, then `evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter --base unsloth/functiongemma-270m-it`. Repeat with one 1–4B alternative. Compare solved rate at h5–h6 on the test split, and recovery rate.

??? example "Hint 3 — a skeleton"
    ```text
    1. On held-out wording, <base + loop> solves <a>; <adapter + loop> solves <b>; at h5–h6 they score <c> vs <d>.
    2. So the harness is worth <…> and fine-tuning adds <…> on top of it — mostly at step <1 | later steps>.
    3. <alternative> scored <…> at <cost: size, latency, VRAM>; I keep <model> because <the horizon column that decided it>.
    4. I would not ship if <the condition from the lesson's ship rule> — today that is <true | false>.
    ```

## Success criteria

- Three rejections, each tied to a spec field.
- One six-step trajectory explained step by step, and the step where rules fail on new wording.
- Solved and recovery rate at two budgets, with one sentence on what the budget protects.
- A patched baseline whose test score rose only because you read the test tickets, and three sentences on why that score no longer measures anything.
- The ablation verdict as two numbers and a sentence.
- (GPU) The four-row table with horizon columns, and a four-sentence memo.

## After you run

The harness turned a keyword guess into a 0.99 investigator on tickets it recognises. The model's job is narrower than it looked: read tickets you didn't write, and read numbers well enough to pick the next screen. Do not check in a trained adapter.

## Lesson link

[Capstone — The On-Call Specialist](../capstone.md)
