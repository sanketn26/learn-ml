# Week 19 — The Job Pipeline

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have a pickle (Week 12) and a legal label (Week 17). sklearn `Pipeline` is an object. This week is the **job**.

---

## 🎯 What you will be able to do

- Draw extract → features → train → **gate** → register → score → monitor as a DAG
- Run `python -m pipelines.train` and get `artifacts/<date>/`, not a file in `/tmp`
- Refuse to promote a model that loses to the dummy or to current prod
- Score tonight’s 80 names from the same `build_features()` training used
- Explain Airflow as cron with retries

!!! think "Think of it like… CI."

    `features.py` is the build. `train.py` is compile. `tests/` + `promote.py` are the required checks. `artifacts/prod` is the release. `score_batch.py` is the nightly deploy. Monitor is the dashboard. Airflow is a fancier `cron`. You already know this system.

## If you already write software

```
CI                              This repo
──────────────────────────      ──────────────────────────────
git commit                      new day’s warehouse partition
build                           pipelines/features.py  (as_of)
unit tests                      tests/test_features.py
package                         artifacts/20240601/model.joblib
required status checks          pipelines/promote.py
deploy                          pipelines/score_batch.py
canary / rollback               keep yesterday’s artifacts/prod
pager                           AUC / precision@80 dropped
```

Week 12’s `Pipeline([prep, model])` is the **binary**. This week is everything around it.

### Picture the DAG

```
          02:00 cron
              │
              ▼
        extract + features(as_of)     ← same function
              │
              ▼
           train.py                   → artifacts/YYYYMMDD/
              │                         model.joblib
              │                         metrics.json
              ▼
          promote.py
           /        \
        fail        pass
         │            │
      keep prod     artifacts/prod = candidate
                      │
                      ▼
                 score_batch.py  → tonight.csv (80 rows)
                      │
                      ▼
                 next week: join labels, write a Slack
```

Nothing in that picture is a vendor. It is four modules:

```
pipelines/
  features.py      as_of → one row per at-risk user
  labels.py        horizon label, censoring
  train.py         writes artifacts/<version>/
  contract.py      validate + predict
  score_batch.py   tonight’s CSV
  promote.py       copy to prod or refuse
tests/
  test_features.py test_labels.py test_contract.py test_gate.py
```

## Run it

From the repo root:

```bash
pytest tests/test_contract.py tests/test_gate.py tests/test_labels.py
python -m pipelines.train --as-of 2024-06-01 --label eventual
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv
```

`train` never writes `prod`. A human or a green gate does. That is the whole difference between a script and a pipeline.

```python
from pipelines.train import train
from pipelines.promote import gate
from pathlib import Path

meta = train("2024-06-01", Path("artifacts"), n=4000)
print(meta["auc"], meta["pr_auc"], meta["dummy_pr_auc"], meta["precision_at_80"])
ok, reason = gate(Path("artifacts") / meta["model_version"], None)
print("promote?", ok, reason)
```

!!! engineer "Engineer mental model"

    Two directories: **candidate** and **prod**. The handler loads `prod`. The training job is not allowed to overwrite it. Same as you do not `scp` onto the live box from your laptop; you promote a build.

## The contract is a test, not a comment

`predict()` and `build_features()` share `FEATURE_COLS`. `validate()` rejects extra keys (that is how `churn_date` and `email` stay out). If training adds a column and forgets the handler, the test in `tests/test_contract.py` fails before Tuesday’s cron.

Training-serving skew that Week 5 could only lecture about:

| Bug | What catches it |
|---|---|
| Train used all-time usage; score used last 30 days | `as_of` in `build_features`, one function |
| Handler reimplemented `log1p` | handler calls `predict()`, no second math |
| New plan type `internal` | `validate` raises; `handle_unknown="ignore"` in the pickle is a last resort |
| Someone put `user_id` in X | `FORBIDDEN` ∩ `FEATURE_COLS` is empty, asserted |

## Batch tonight vs `/predict`

```
Batch (ship this first)          Online (later)
score everyone at 2am            score this payload now
CSV / Slack to CS                POST /predict
failure = a late email           failure = a 500 on a request
same artifact                    same artifact
```

Week 12 was right: you may ship the batch list. You may not ship a public HTTP API until `contract.py` is imported by the handler, not copy-pasted into FastAPI.

## Monitor is last week’s labels

Drift histograms (Week 12) are a smoke alarm. The actual page:

1. Take last week’s `tonight.csv`
2. Now that 30 days have passed, join the horizon label
3. Print precision@80 vs what `metrics.json` promised
4. If it fell off a cliff, do **not** auto-promote tomorrow’s train

That is a 20-line job. It is more valuable than a feature store.

!!! warning "Watch out"

    - Retraining daily on a 6.7% event is how you overfit the last noisy week. Weekly is a default.
    - Auto-promote without a gate is `main` pushing to prod on red CI.
    - Two copies of feature math is two products. You will not notice until a whale gets a 0.0.

!!! success "Ship / don’t ship"

    Ship a cron, a candidate directory, a gate, and a CSV. Do not ship Kubeflow so you can say “we have a platform.” Do not let `train.py` overwrite `prod`. Do not add Airflow until a cron file is boring.

## ✍️ Exercise

[Exercises](exercises/week-19.md). Run `pytest tests/` from the repo root.

## 🤔 Reflection

1. Who is allowed to write `artifacts/prod`? Who is allowed to read it?
2. Tomorrow’s PR-AUC is 0.01 worse than prod. Promote? Wait? Page?
3. Why is “we will clean the features up in the handler” a pipeline bug, not a style comment?

## 🔗 Next week

You are on-call. A bad join, a leaked label, a silent NaN. Then a ticket bot that uses this score as a *tool*, not as a personality.
