# Exercises — Week 16 — The Job Pipeline

## What you are building

A candidate directory, a promote gate, proof that train does not write prod, and a five-line cron.

## Predict before you run

1. After two trains with the same `--as-of`, did `artifacts/prod` change?
2. If you raise dummy PR-AUC above the candidate, does promote refuse?
3. Who is allowed to write `artifacts/prod`?

## Task

Work in `starter.py`. Run from the repo root:

```bash
pytest tests/
python exercises/ml/week-16/starter.py
```

Full pipeline (after you trust the starter):

```bash
python -m pipelines.train --as-of 2024-06-01 --n 8000 --label eventual
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv
head tonight.csv
```

**1. Gate.** After training, open `artifacts/20240601/metrics.json`. Confirm `pr_auc > dummy_pr_auc`. If you temporarily set the dummy higher in a scratch copy of `promote.gate`, the promote must refuse.

**2. Train does not write prod.** Run train twice with the same `--as-of`. `artifacts/prod` must change only after `promote`.

**3. One function.** In `tests/test_features.py`, add (or just read) the assertion that `FEATURE_COLS` never intersects `FORBIDDEN`.

**4. Cron.** Write a five-line shell script you would hang on a weekly timer: pytest, train, promote, score. Do not add Airflow.

## Success criteria

- metrics.json beats dummy.
- Promote refuses a worse dummy.
- Prod unchanged across two trains.
- Five-line cron, no Airflow.

## Debugging clues

- Horizon labels can starve positives — this course defaults to `eventual`.
- Auto-promote without a gate is red CI merging to main.
- Two copies of feature math is two products.

## After you run

The job is train → gate → prod dir → tonight's CSV. Kubeflow is not the week.

## Lesson link

[Week 16 — The Job Pipeline](../week-16.md)
