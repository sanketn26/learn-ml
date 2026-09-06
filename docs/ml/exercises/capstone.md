---
description: Capstone exercises rejecting a broken tool call, adding a new scenario to the evaluation harness, and writing a ship or don't-ship baseline memo.
---

# Exercises — Capstone — Reliable Coding Specialist

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

This runs Phases 1, 2, 4, and 5 on CPU with no API key. Phase 3 GPU train is optional — dry-run the scaffold first (commands above).

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
   [Capstone Phase 3](../capstone.md#phase-3-fine-tune-scaffold-first-colab-for-the-gpu)).
4. Export to GGUF, download it, and run it locally with `llama.cpp` or `ollama`.
5. Point `capstone.evaluate.specialist_call` at your local model instead of
   the golden-answer placeholder (or pass `--adapter` to
   `evaluate_adapter.py`) and rerun. The accuracy gap vs. the placeholder is
   your real specialization gap — report it.

## Lesson link

[Capstone — Reliable Coding Specialist](../capstone.md)
