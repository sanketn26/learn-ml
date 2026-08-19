# Exercises — Capstone: Reliable Coding Specialist

Do these after reading the [Capstone](../capstone.md). Work in
`exercises/ml/capstone/starter.py` and the `capstone/` package at the repo
root. Run from the repo root:

```bash
python exercises/ml/capstone/starter.py
python -m capstone.evaluate
pytest tests/test_capstone.py
```

This runs entirely on CPU, no API key required — it exercises Phases 1, 2, 4,
and 5 (tool contract, trajectory generation, reliability checks, evaluation
harness). Phase 3 (the actual fine-tune) is a separate, GPU step — see the
bottom of this page.

**1. Break the contract on purpose.** In a Python shell, call
`capstone.reliability.validate_call({"name": "delete_repo", "arguments": {}})`
and `validate_call({"name": "explain_error", "arguments": {}})`. Read the
`RejectedCall` messages. Which check fired for each?

**2. Add a seventh scenario.** In `capstone/scenarios.py`, add one more
`Scenario` for a tool of your choice, then add the matching branch to
`capstone.teacher.golden_call`. Rerun `python -m capstone.evaluate` — did the
specialist's placeholder accuracy stay at 1.0? (It should — that's the point
of `validate_call()` running inside `build_trajectory`.)

**3. Make the baseline worse, then better.** In `capstone/baseline.py`, change
`_WEIGHTS` to raise `hallucinated_tool` to 0.4. Rerun `python -m capstone.evaluate`.
Now imagine you've fine-tuned a real model and its accuracy landed at 0.75, not
1.0 — write two sentences on whether you'd ship it, referencing the "ship /
don't ship" rule from the lesson.

**4. Base-model memo.** Four sentences to a teammate: would you start Phase 3
with FunctionGemma or the alternative you'd pick, and what specific Phase 5
metric would change your mind?

## Phase 3 — the GPU step (outside the repo's normal laptop flow)

1. Generate a larger trajectory set: extend `capstone/scenarios.py` past the
   six teaching examples, or wire a real teacher model behind
   `capstone/teacher.py` using the schema from `capstone.tools.tool_schema()`.
2. Open a Colab notebook with a T4/L4 GPU runtime. Install `unsloth`, `peft`,
   `trl`, `bitsandbytes` (see `requirements-capstone.txt` for pinned
   dependencies — install it in Colab, not on your laptop).
3. Fine-tune FunctionGemma-270M (or your chosen alternative) on
   `capstone/data/train.jsonl`, following the sketch in the
   [Capstone Phase 3](../capstone.md#phase-3-fine-tune-on-colab) section.
4. Export to GGUF, download it, and run it locally with `llama.cpp` or `ollama`.
5. Point `capstone.evaluate.specialist_call` at your local model instead of
   the golden-answer placeholder and rerun `python -m capstone.evaluate`. The
   accuracy gap vs. the placeholder is your real specialization gap — report it.
