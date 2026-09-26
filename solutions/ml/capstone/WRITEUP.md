# Capstone — recovery writeup

Lesson: [docs/ml/capstone.md](../../../docs/ml/capstone.md)
Exercise: [docs/ml/exercises/capstone.md](../../../docs/ml/exercises/capstone.md)

!!! warning "Do not open `solution.py` until you are stuck after the hints"

    Work in `exercises/ml/capstone/starter.py` first. Tasks 1–5 are
    CPU-only and offline. Task 6 is the GPU step.

## Hint 1

??? tip "Hint 1"

    A command is only as narrow as its spec. `feature_column`, `user_id`,
    `quote`, and `steps` are the arg types that check meaning against the
    investigation so far. The harness is what makes a six-step ticket six
    one-step decisions: read the result, pick the next screen.

## Hint 2

??? tip "Hint 2"

    `run_loop(teacher_policy(case), case)` solves every case; render its
    state to see the runbook. `rules_policy` is the same runbook after a
    keyword guess, so where it fails, it fails at step 1. A rejection costs
    a step — a budget equal to the teacher's step count leaves no room to
    recover.

## Debugging clues

??? warning "Debugging clues"

    - The first `check_threshold` trains the production model into
      `artifacts/capstone-commands/`. Delete that folder to retrain.
    - A `user_id` rejected with "does not appear in any earlier result" means
      the policy invented it — use `example_user_id` from `compare_nights`.
    - The teacher row is a ceiling: it is told the first command.
    - Keywords read off the test tickets make the test score meaningless.
    - Do not `import unsloth` on the laptop; the dry-run scaffold never does.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/capstone/solution.py
```

It covers tasks 1–5 and prints the ablation. It does **not** ship a trained
adapter.

Task 6 (fine-tune):

```bash
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
python capstone/finetune/evaluate_adapter.py --dry-run
```

Real training needs a GPU (Colab T4/L4). See
[capstone/finetune/README.md](../../../capstone/finetune/README.md).

## Why this decision

The harness takes a keyword baseline from 0.54 to 0.99 on tickets it
recognises, with no model at all — so the spec and the loop come first.
What's left for weights is narrow: read tickets nobody wrote a keyword for,
and read results well enough to pick the next screen. A specialist earns its
place only by beating its own base model *inside the same harness*, on
held-out wording, at the long horizons.
