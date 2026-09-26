---
description: Wire feature extraction, training, gating, and scoring into an automated ML pipeline job, treating Airflow as cron with retries.
---

# Week 16 — The Job Pipeline

Ana's rule, in writing this time: nothing writes to `artifacts/prod` unless it beats the dummy and beats current prod, automatically, every single night. You are the one wiring that gate. You're also on-call starting next week.

??? note "Course details"

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
python -m pipelines.train --as-of 2024-06-01
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-07-01 --artifact artifacts/prod --out tonight.csv
```

`train` backtests (Week 15): it learns on the snapshot `--horizon-days` before `--as-of` (default 30), labelled with what happened by `--as-of`, then scores the `--as-of` snapshot against the next 30 days — the question Priya asked. The horizon, both snapshot dates, the date its labels are known by, and a bootstrap interval on precision@80 go into `metrics.json`, so you do not lie about which question you shipped or how sure you were.

That is why the score date is **July 1**, not June 1. The backtest as of June 1 was graded on who churned by July 1; on the morning of June 1 nobody knows that yet. A job has three dates, and they are never the same day:

```
 train learns on     train is graded on            the job runs and scores
 2024-05-02 ────────► 2024-06-01 ─── 30 days ───► 2024-07-01
 (as_of − horizon)    (as_of)     labels mature    (score date = as_of + horizon)
```

`python -m pipelines.job --score-date 2024-07-01` does all four steps with the dates derived from the one you know: today. `score_batch` refuses a score date earlier than the model's `labels_known_by`.

`train` never writes `prod`. A human or a green gate does. That is the whole difference between a script and a pipeline.

```python
from pathlib import Path

from pipelines.promote import gate
from pipelines.train import train

meta = train("2024-06-01", Path("artifacts"))
print(meta["auc"], meta["pr_auc"], meta["dummy_pr_auc"], meta["precision_at_80"], meta["base_rate"])
ok, reason = gate(Path("artifacts") / meta["model_version"], Path("artifacts") / "prod")
print("promote?", ok, reason)
```

!!! warning "Watch out — compare two models on the same customers"

    prod's `metrics.json` holds its PR-AUC from *its* backtest: a different month, a different base rate, a different mix of easy and hard accounts. PR-AUC moves with all three. A candidate that "beats 0.05" might just have been graded in a month when more people churned. So `gate` loads prod's pipeline and re-scores it on the candidate's matured holdout, then compares the two numbers from the same rows. The reason string says so: `ok: PR-AUC 0.0512 vs prod 0.0498 (same holdout, 27,935 customers)`.

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

## Slice report before you promote

The gate asks one question about the whole population: does the candidate beat the dummy? Before a human promotes, ask it again **per slice** — plan, tenure band, region, company size, whatever your business is organized by. One table:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipelines.contract import load_artifact
from pipelines.features import FEATURE_COLS
from pipelines.split import snapshot_split

art = load_artifact(Path("artifacts/prod"))
_, _, test, y = snapshot_split("2024-06-01")
test = test.assign(score=art["pipeline"].predict_proba(test[FEATURE_COLS])[:, 1], y=y.to_numpy())
# Highest score first; user_id breaks ties so the list is the same on every run.
order = np.lexsort((test["user_id"].to_numpy(), -test["score"].to_numpy()))
test["on_list"] = False
test.iloc[order[:80], test.columns.get_loc("on_list")] = True
test["tenure_band"] = pd.cut(test["tenure_so_far"], [-1, 90, 365, 10_000], labels=["<90d", "90–365d", ">1y"])

for col in ["plan_type", "tenure_band"]:
    rows = []
    for key, g in test.groupby(col, observed=True):
        rows.append({
            col: key,
            "customers": len(g),
            "churn rate": g["y"].mean(),
            "share of list": g["on_list"].sum() / 80,
            "AUC in slice": roc_auc_score(g["y"], g["score"]) if g["y"].nunique() > 1 else np.nan,
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")
```

On this model the list is about 85% `free` and over 90% customers younger than 90 days. No `pro` or `enterprise` account makes it, and inside `enterprise` the model barely ranks at all (AUC near 0.55). None of that is a bug — free, new accounts really do churn most. But it is a *product decision hiding inside a model*: Priya's team will spend every call on accounts worth $0 of MRR, and nobody will ever call a wobbling enterprise customer. That is Helen's call to make, with this table in front of her, not a default the model makes for her. (The [scenario bank](capstone-scenarios.md) shows briefs that make it explicitly.)

!!! warning "Watch out — this table is also your fairness audit"

    CloudWave's customers are companies, so the slices are plans and tenure. When your rows are *people*, the same table by age band, region, language, or disability status is the fairness check. Those attributes stay **out of `X`** (Week 8's PII fence) and **in the audit** — you cannot measure a gap on a column you refused to keep. Look for a slice where the model is much worse at ranking (AUC in slice), or where one group absorbs all the false alarms, and decide *before* promoting whether that is acceptable.

## Monitor is last week’s labels

Drift histograms (Week 15) are a smoke alarm. The actual page:

1. Take last week’s `tonight.csv`
2. Now that 30 days have passed, join the horizon label
3. Print precision@80 vs the **interval** `metrics.json` promised (`precision_at_80_ci95`)
4. One week below the interval is a yellow light — 80 calls hold only a handful of churners (Week 11). Several weeks below it is a page: do **not** auto-promote tomorrow’s train

That is a 15-line job. It is more valuable than a feature store.

```python
import pandas as pd

from pipelines.features import build_features
from pipelines.labels import label_churn_in_horizon

as_of = "2024-06-01"  # the night we scored
tonight = pd.read_csv("tonight.csv")
frame = build_features(as_of=as_of, n=None, at_risk_only=True)
frame = frame.assign(y=label_churn_in_horizon(frame, as_of, horizon_days=30))  # same horizon as metrics.json
joined = tonight.merge(frame[["user_id", "y"]], on="user_id", how="left")
knowable = joined.dropna(subset=["y"])
print("n flagged", len(tonight), "with labels", len(knowable))
print("precision@80", float(knowable["y"].mean()) if len(knowable) else "still censored")
# compare to metrics.json["precision_at_80_ci95"], not the point estimate
```

`score_batch` already `validate`s every row it scores. You do not need a second loop.

!!! warning "Watch out"

    - Retraining daily on a ~2% 30-day event (whatever `base_rate` you wrote in `metrics.json`) is how you overfit the last noisy week. Weekly is a default.
    - Once the list ships, called customers are not clean labels (Week 11). Log who was called; monitor precision on the randomized holdout, or a list that *works* will look like a list that broke.
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

The gate is wired. The job runs tonight. Tomorrow morning you're the one who finds out if it held: a bad join, a leaked label, a silent NaN — three ways this can quietly go wrong in prod, and [Week 17](week-17.md) is where you meet all three. You're on-call.
