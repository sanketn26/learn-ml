# Exercises — Week 16 — The Job Pipeline

Do these after reading [Week 16](../week-16.md).

```bash
pytest tests/
python -m pipelines.train --as-of 2024-06-01 --n 8000 --label eventual
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv
head tonight.csv
```

**1. Gate.** After training, open `artifacts/20240601/metrics.json`. Confirm `pr_auc > dummy_pr_auc`. If you temporarily set the dummy higher in a scratch copy of `promote.gate`, the promote must refuse.

**2. Train does not write prod.** Run train twice with the same `--as-of`. `artifacts/prod` must change only after `promote`.

**3. One function.** In `tests/test_features.py`, add (or just read) the assertion that `FEATURE_COLS` never intersects `FORBIDDEN`.

**4. Cron.** Write a five-line shell script you would hang on a weekly timer: pytest, train, promote, score. Do not add Airflow.
