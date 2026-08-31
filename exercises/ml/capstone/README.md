# Exercise — Capstone — Reliable Coding Specialist

## What you are building

A broken tool call, a seventh scenario + teacher branch, a worse-then-better baseline memo, and a four-sentence base-model choice. Phase 3 is the finetune scaffold, not a dumped checkpoint.

## Predict before you run

1. Which `RejectedCall` fires for `delete_repo` vs empty `explain_error`?
2. If you add a seventh scenario and a matching `golden_call`, does placeholder accuracy stay 1.0?
3. Would you ship a real adapter at 0.75 accuracy vs this baseline?

## Task

Work in `exercises/ml/capstone/starter.py` and the `capstone/` package. Run from the repo root:

```bash
python exercises/ml/capstone/starter.py
python -m capstone.evaluate
pytest tests/test_capstone.py
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
```

This runs Phases 1, 2, 4, and 5 on CPU with no API key. Phase 3 GPU train is optional — see [docs/ml/exercises/capstone.md](../../../docs/ml/exercises/capstone.md).

**1. Break the contract on purpose.** Call `validate_call({"name": "delete_repo", "arguments": {}})` and `validate_call({"name": "explain_error", "arguments": {}})`. Which check fired for each?

**2. Add a seventh scenario.** In `capstone/scenarios.py`, add one more `Scenario`, then the matching branch in `golden_call`. Rerun `python -m capstone.evaluate` — did the specialist placeholder stay at 1.0?

**3. Make the baseline worse, then better.** Raise `hallucinated_tool` in `baseline._WEIGHTS` to 0.4. Rerun evaluate. Then write two sentences: would you ship a real model at 0.75 accuracy?

**4. Base-model memo.** Four sentences: FunctionGemma vs an alternative, and which Phase 5 metric would change your mind.

## Success criteria

- Both RejectedCall messages named.
- Seventh scenario scored.
- Ship/don't-ship sentences reference the lesson rule.
- `--dry-run` finetune scripts exit 0 without a GPU.

## Debugging clues

- Placeholder 100% is a ceiling, not a trained model.
- Teacher rows must pass `validate_call` before they hit jsonl.
- Do not `import unsloth` on the laptop.

## After you run

A specialist is a narrow IAM policy plus a lint rule. Phase 3 is `capstone/finetune/`. Do not check in a trained adapter.

## Phase 3

Laptop: `python capstone/finetune/prepare_data.py --dry-run` (no GPU). Real
train: Colab T4/L4 or any ~12 GB GPU — see
[docs/ml/exercises/capstone.md](../../../docs/ml/exercises/capstone.md).

## Lesson link

[Capstone — Reliable Coding Specialist](../../../docs/ml/capstone.md)
