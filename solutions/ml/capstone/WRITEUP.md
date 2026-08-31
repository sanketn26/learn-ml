# Capstone — recovery writeup

Lesson: [docs/ml/capstone.md](../../../docs/ml/capstone.md)
Exercise: [docs/ml/exercises/capstone.md](../../../docs/ml/exercises/capstone.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/capstone/starter.py` first. Phases 1–2 and 4–5 are
    CPU-only. Phase 3 is the GPU step.

## Hint 1

??? tip "Hint 1"

    The tool surface is a contract, same as Week 15's `validate()`. A call
    that is not in `TOOLS` or is missing a required arg is a
    `RejectedCall` — reject, don't repair. The teacher is *you* for the six
    synthetic scenarios. A seventh scenario must get a `golden_call` branch
    or `write_splits` will raise.

## Hint 2

??? tip "Hint 2"

    `validate_call({"name": "delete_repo", ...})` fires *unknown tool*.
    `explain_error` with `{}` fires *missing required args*. Raising
    `hallucinated_tool` weight in `baseline._WEIGHTS` should drop baseline
    accuracy. Phase 3 does not live in this `solution.py` — use
    `capstone/finetune/` (`prepare_data.py --dry-run`, then Colab or
    `train_lora.py` on a GPU).

## Debugging clues

??? warning "Debugging clues"

    - `specialist_call` is a placeholder that returns the golden answer.
      100% accuracy is a ceiling, not a trained model.
    - Teacher output that fails `validate_call` must not be written to jsonl.
    - A wide tool surface is a wide attack surface (`s6_injection`).
    - Do not `import unsloth` on the laptop; the dry-run scaffold will skip it.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/capstone/solution.py
```

It covers Phases 1–2 (tools, teacher, `validate_call`) and the evaluation
harness. It does **not** dump a trained adapter.

Phase 3 (fine-tune):

```bash
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
python capstone/finetune/evaluate_adapter.py --dry-run
```

Real training needs a GPU (Colab T4/L4 is the supported path). See
[capstone/finetune/README.md](../../../capstone/finetune/README.md).

## Why this decision

A specialist beats a generalist here because the decision space is five
tools, not because 270M parameters are wiser. Reliability is
`validate_call` in front of the model, the same way `validate()` sits in
front of the pickle. Training on a trajectory that failed its own schema
teaches confident garbage.
