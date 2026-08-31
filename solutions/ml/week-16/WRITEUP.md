# Week 16 — recovery writeup

Lesson: [docs/ml/week-16.md](../../../docs/ml/week-16.md)
Exercise: [docs/ml/exercises/week-16.md](../../../docs/ml/exercises/week-16.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-16/starter.py` first.

## Hint 1

??? tip "Hint 1"

    `train.py` writes `artifacts/<date>/`. Only `promote.py` may write
    `artifacts/prod`. The gate is a function of `metrics.json`, not a
    feeling. The cron is five lines: test, train, promote, score.

## Hint 2

??? tip "Hint 2"

    After `train(...)`, open `metrics.json` and compare `pr_auc` to
    `dummy_pr_auc`. Call `pipelines.promote.gate` with a scratch metrics
    file that sets dummy higher — it must refuse. `FEATURE_COLS` ∩
    `FORBIDDEN` is asserted in `tests/test_features.py`.

## Debugging clues

??? warning "Debugging clues"

    - Running train twice with the same `--as-of` *should* overwrite the
      candidate dir. Prod must stay put until promote.
    - If `artifacts/prod` is missing, gate still requires beating the dummy.
    - Horizon labels can starve the train set — this course defaults to
      `label=eventual` and says so in metrics.
    - Auto-promote on a worse PR-AUC is `main` pushing to prod on red CI.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-16/solution.py
```

It trains a small candidate under `artifacts/solution-week16/` so it does
not clobber a prod pickle you already promoted.

```python
ok, reason = gate(candidate, prod)
```

## Why this decision

Training is allowed to be sloppy and frequent. Production is a copy that
survived a numeric gate. Splitting those directories is the whole job:
tomorrow's worse model stays a candidate, tonight's CSV still comes from
`artifacts/prod`.
