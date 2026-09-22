# Exercise — Capstone — Reliable Coding Specialist

## What you are building

A broken tool call, a seventh scenario + teacher branch, a worse-then-better baseline memo, and a four-sentence base-model choice. Phase 3 is the finetune scaffold, not a dumped checkpoint.

## Predict before you run

1. Which `RejectedCall` fires for `delete_repo` vs empty `explain_error`?
2. If you add a seventh scenario and a matching `golden_call`, does placeholder accuracy stay 1.0?
3. Would you ship a real adapter at 0.75 accuracy vs this baseline?

## Before you start

- Do not `import unsloth` (or `peft`, `trl`, `bitsandbytes`) on the laptop. Those are Phase 3 only, in Colab.
- The "specialist" in `capstone.evaluate` is a placeholder that returns the golden answer. 100% is a ceiling, not a trained model.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `exercises/ml/capstone/starter.py` and the `capstone/` package. Run from the repo root:

```bash
python exercises/ml/capstone/starter.py
python -m capstone.evaluate
pytest tests/test_capstone.py
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
```

This runs Phases 1, 2, 4, and 5 on CPU with no API key. Phase 3 GPU train is optional — dry-run the scaffold first (commands above).

**1. Break the contract on purpose.** Call `validate_call({"name": "delete_repo", "arguments": {}})` and `validate_call({"name": "explain_error", "arguments": {}})`. Which check fired for each?

<details>
<summary>Hint 1 — a nudge</summary>

`validate_call` is a list of checks run in order, and it stops at the first failure. Which check would a tool that doesn't exist fail before it ever gets to its arguments?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Wrap each call in `try` / `except RejectedCall as exc` and print `exc`. Then open `capstone/reliability.py` and match each message to the line that raised it.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
from capstone.reliability import RejectedCall, validate_call

for call in ({"name": "delete_repo", "arguments": {}}, {"name": "explain_error", "arguments": {}}):
    try:
        validate_call(call)
    except RejectedCall as exc:
        print(f"{call['name']:<14} → {exc}")
```

</details>

**2. Add a seventh scenario.** In `capstone/scenarios.py`, add one more `Scenario`, then the matching branch in `golden_call`. Rerun `python -m capstone.evaluate` — did the specialist placeholder stay at 1.0?

<details>
<summary>Hint 1 — a nudge</summary>

The "specialist" today is `golden_call` itself. What would have to be true of your new scenario for it to score anything other than 100%?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Pick an incident you already know — a Week 15 contract error, a Week 16 gate refusal — and an existing tool. `golden_call` has a branch per tool, so a new scenario on an existing tool needs no new branch; a new *tool* needs `capstone/tools.py` too. Teacher rows must pass `validate_call` before they reach the jsonl — `build_trajectory` checks that for you. Try it in memory before you edit the file.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
from capstone.evaluate import run, specialist_call, summarize
from capstone.scenarios import SCENARIOS, Scenario
from capstone.teacher import build_trajectory

s7 = Scenario(
    id="s7_extra_payload_key",
    expect_tool="explain_error",
    context="Week 15 — a payload with user_id hits the contract.",
    input_text="Traceback (most recent call last):\n  ...\nValueError: unknown keys: ['user_id']\n",
)
print(build_trajectory(s7)["tool_call"])
print(summarize(run(specialist_call, SCENARIOS + [s7])))
```
Moving it into `scenarios.py` — and explaining the accuracy you see — is yours.

</details>

**3. Make the baseline worse, then better.** Raise `hallucinated_tool` in `baseline._WEIGHTS` to 0.4. Rerun evaluate. Then write two sentences: would you ship a real model at 0.75 accuracy?

<details>
<summary>Hint 1 — a nudge</summary>

Accuracy is one number over six scenarios. Which *kind* of miss is a hallucinated tool, and would you accept one in production at any accuracy?

</details>

<details>
<summary>Hint 2 — the approach</summary>

`_WEIGHTS` lines up with `_FAILURE_MODES`; `hallucinated_tool` is index 2. You can patch the list in memory to experiment, then print `summarize(...)["by_outcome"]` — not just accuracy. The lesson's ship rule names the outcome that is disqualifying on its own.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import capstone.baseline as baseline

def baseline_summary() -> dict:
    return summarize(run(lambda s: baseline.general_model_call(s, seed=0)))

print("before:", baseline_summary())
baseline._WEIGHTS[baseline._FAILURE_MODES.index("hallucinated_tool")] = 0.4
print("after: ", baseline_summary())
```

</details>

**4. Base-model memo.** Four sentences: FunctionGemma vs an alternative, and which Phase 5 metric would change your mind.

<details>
<summary>Hint 1 — a nudge</summary>

A base model is a dependency. What would you check before adding any dependency to a service with a latency budget and a laptop-sized deploy target?

</details>

<details>
<summary>Hint 2 — the approach</summary>

One sentence each: what FunctionGemma is good for here (size, native function-calling), one concrete alternative and its trade-off, the Phase 5 metric you'd watch (outcome counts, not just accuracy), and the threshold on it that would make you switch.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
1. FunctionGemma-270M fits because <size / tool-call format / local deploy>.
2. <alternative> would <gain> at the cost of <cost>.
3. The Phase 5 number I'd watch is <outcome from by_outcome>, not headline accuracy.
4. If <that outcome> exceeds <threshold> after fine-tuning, I switch to <alternative>.
```

</details>

## Success criteria

- Both RejectedCall messages named.
- Seventh scenario scored.
- Ship/don't-ship sentences reference the lesson rule.
- `--dry-run` finetune scripts exit 0 without a GPU.

## After you run

A specialist is a narrow IAM policy plus a lint rule. Phase 3 is `capstone/finetune/`. Do not check in a trained adapter.

## Phase 3 — the GPU step (outside the repo's normal laptop flow)

The scripts live in `capstone/finetune/`. Colab is still an option; it is no
longer the only description of Phase 3.

1. Generate a larger trajectory set: extend `capstone/scenarios.py` past the
   six teaching examples, or wire a real teacher model behind
   `capstone/teacher.py` using the schema from `capstone.tools.tool_schema()`.
   Format and validate with `python capstone/finetune/prepare_data.py`.
2. Open a Colab notebook with a T4/L4 GPU runtime **or** any 12 GB GPU box.
   Install `unsloth`, `peft`, `trl`, `bitsandbytes` (see
   `requirements-capstone.txt` — install it in Colab, not on your laptop).
3. Fine-tune `google/functiongemma-270m-it` (or your chosen alternative) with
   `python capstone/finetune/train_lora.py` (see
   [Capstone Phase 3](../../../docs/ml/capstone.md#phase-3-fine-tune-scaffold-first-colab-for-the-gpu)).
4. Export to GGUF, download it, and run it locally with `llama.cpp` or `ollama`.
5. Point `capstone.evaluate.specialist_call` at your local model instead of
   the golden-answer placeholder (or pass `--adapter` to
   `evaluate_adapter.py`) and rerun. The accuracy gap vs. the placeholder is
   your real specialization gap — report it.

## Lesson link

[Capstone — Reliable Coding Specialist](../../../docs/ml/capstone.md)
