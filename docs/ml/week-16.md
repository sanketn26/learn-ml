---
description: Wire feature extraction, training, gating, and scoring into an automated ML pipeline job, treating Airflow as cron with retries.
---

# Week 16 — The Job Pipeline

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have a pickle (Week 15) and a legal label (Week 8). sklearn `Pipeline` is an object. This week is the **job**.

---

## 🎯 What you will be able to do

- Draw extract → features → train → **gate** → promote → score → monitor as a DAG
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

Week 15’s `Pipeline([prep, model])` is the **binary**. This week is everything around it.

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
python -m pipelines.train --as-of 2024-06-01 --n 8000 --label eventual
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv
```

`--label eventual` (default) is “did they cancel *after* `as_of`.” This fixture only has tens of 30-day events, so that is the question the file can supervise. `--label horizon` is the product question (cancel in 30 days). It will often refuse to train: one class in the fold. Both write `"label"` into `metrics.json` so you do not lie about which one you shipped.

`train` never writes `prod`. A human or a green gate does. That is the whole difference between a script and a pipeline.

```python
from pathlib import Path

from pipelines.promote import gate
from pipelines.train import train

meta = train("2024-06-01", Path("artifacts"), n=8000, label="eventual")
print(meta["auc"], meta["pr_auc"], meta["dummy_pr_auc"], meta["precision_at_80"], meta["base_rate"])
ok, reason = gate(Path("artifacts") / meta["model_version"], Path("artifacts") / "prod")
print("promote?", ok, reason)
```

!!! engineer "Engineer mental model"

    Two directories: **candidate** and **prod**. The handler loads `prod`. The training job is not allowed to overwrite it. Same as you do not `scp` onto the live box from your laptop; you promote a build.

## The contract is a test, not a comment

`predict()` and `build_features()` share `FEATURE_COLS`. `validate()` rejects extra keys (that is how `churn_date` and `email` stay out). If training adds a column and forgets the handler, the test in `tests/test_contract.py` fails before Tuesday’s cron.

Training-serving skew that Week 6 could only lecture about:

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

Week 15 was right: you may ship the batch list. You may not ship a public HTTP API until `contract.py` is imported by the handler, not copy-pasted into FastAPI.

## Monitor is last week’s labels

Drift histograms (Week 15) are a smoke alarm. The actual page:

1. Take last week’s `tonight.csv`
2. Now that 30 days have passed, join the horizon label
3. Print precision@80 vs what `metrics.json` promised
4. If it fell off a cliff, do **not** auto-promote tomorrow’s train

That is a 15-line job. It is more valuable than a feature store.

```python
import pandas as pd

from pipelines.features import build_features
from pipelines.labels import label_churn_in_horizon

as_of = "2024-06-01"  # the night we scored
tonight = pd.read_csv("tonight.csv")
frame = build_features(as_of=as_of, n=None, at_risk_only=True)
frame = frame.assign(y=label_churn_in_horizon(frame, as_of))
joined = tonight.merge(frame[["user_id", "y"]], on="user_id", how="left")
knowable = joined.dropna(subset=["y"])
print("n flagged", len(tonight), "with labels", len(knowable))
print("precision@80", float(knowable["y"].mean()) if len(knowable) else "still censored")
# compare to metrics.json["precision_at_80"] and metrics.json["base_rate"]
```

`score_batch` already `validate`s every row it scores. You do not need a second loop.

!!! warning "Watch out"

    - Retraining daily on a ~0.1% 30-day event (or whatever `base_rate` you wrote in `metrics.json`) is how you overfit the last noisy week. Weekly is a default.
    - Auto-promote without a gate is `main` pushing to prod on red CI.
    - Two copies of feature math is two products. You will not notice until a whale gets a 0.0.

!!! success "Ship / don’t ship"

    Ship a cron, a candidate directory, a gate, and a CSV. Do not ship Kubeflow so you can say “we have a platform.” Do not let `train.py` overwrite `prod`. Do not add Airflow until a cron file is boring.

## ✍️ Exercise

[Exercises](exercises/week-16.md). Run `pytest tests/` from the repo root.

## 🤔 Reflection

1. Who is allowed to write `artifacts/prod`? Who is allowed to read it?
2. Tomorrow’s PR-AUC is 0.01 worse than prod. Promote? Wait? Page?
3. Why is “we will clean the features up in the handler” a pipeline bug, not a style comment?

## Before you leave

Try one [self-check](self-checks.md#week-16-the-job) (Predict / Diagnose / Choose / Defend). Write the answer before you open the block.

## 🔗 Next week

You are on-call. A bad join, a leaked label, a silent NaN. Then a ticket bot that uses this score as a *tool*, not as a personality.
