---
description: The job-path capstone — take a CloudWave churn score from as-of features to a gated nightly list, choose its threshold from a real capacity budget, and diagnose a production incident.
---

# Capstone — Ship the Churn Score

Quarter-end. Helen wants one line in the board pack: *we have a churn score, and it runs every week without anyone babysitting it.* Priya wants her 80 names on Monday morning, every Monday. Ana wants to go on holiday. None of them wants another notebook. Everything this needs, you have already built in pieces across seventeen weeks — the capstone is the first time they all have to hold at once, and the first time something breaks that you didn't write.

??? note "Course details"

    **Course:** Applied ML Foundations for SaaS Analytics
    **Who this is for:** Anyone who finished the required track (Weeks 0–17). CPU-only, no API key, about an afternoon.

---

## 🎯 What you will be able to do

- Compose `build_features`, `snapshot_split`, `train`, `promote`, and `score_batch` into one weekly job without re-deriving any of them
- Choose a label horizon from what the data can supervise, and say what it cost
- Set a threshold from a stated capacity budget, and write it into the artifact
- Say honestly what an 80-name list is worth when the event is rare
- Diagnose a production incident that passes every contract check, and name the test that would have caught it

!!! think "Think of it like… a release train with seven gates."

    You don't ship a service by writing the handler. You ship it when the schema migration, the tests, the canary, the rollout, and the pager are all in place — and then something upstream changes on a Tuesday. Each gate below is a check you already wrote in some week. The capstone is the train.

## The picture

```
 as_of ──► 1 features ──► 2 label ──► 3 train ──► 4 threshold ──► 5 contract ──► 6 promote + score ──► tonight.csv
 06-01     (Week 3/6)     (Week 8)    (Week 15)   (Week 11)       (Week 15)      (Week 16)   07-01        │
                                                                                                           ▼
                                                                    two weeks later: 7 incident (Week 17) ── Priya: "who are these people?"
```

Two dates, on purpose. The backtest is **as of June 1**: learn on May 2's customers, grade on who of June 1's customers left by July 1. So **July 1** is the first morning that model could exist, and the first list it scores.

Steps 1–6 are the job. Step 7 is why you built it that way.

```python
from pathlib import Path

from capstone_ship.briefs import RETENTION_DESK, judge, select, threshold_for
from pipelines.contract import load_artifact
from pipelines.features import FEATURE_COLS
from pipelines.promote import promote
from pipelines.score_batch import score_batch
from pipelines.split import snapshot_split
from pipelines.train import train

train_df, y_train, test_df, y_test = snapshot_split("2024-06-01", horizon_days=30)       # 2
meta = train("2024-06-01", Path("artifacts/demo"), horizon_days=30)              # 3 (builds 1 inside)
candidate = Path("artifacts/demo") / meta["model_version"]
scores = load_artifact(candidate)["pipeline"].predict_proba(test_df[FEATURE_COLS])[:, 1]
picked = select(test_df, scores, RETENTION_DESK)                                          # 4
print(judge(picked, test_df, y_test), "threshold", round(threshold_for(picked), 4))
promote(candidate, Path("artifacts/demo/prod"))                                            # 6 (gate inside)
print(score_batch("2024-07-01", Path("artifacts/demo/prod")).head(3))                     # 5 validates every row
```

Nine calls, no new modelling code. If any step needs more than a few lines, it's re-deriving something a week already owns.

## 1 — Features as of a morning

`build_features(as_of, n=None)`: one row per customer at risk on that date, every aggregate cut at `as_of`. The capstone's only addition is asserting it — unique `user_id`, no signups after `as_of`, no lifetime columns in `FEATURE_COLS`. [Week 3](week-03.md) and [Week 6](week-06.md) own the why.

## 2 — A label with a horizon you can defend

Priya's real question is *who leaves in the next 30 days?* Check that the file can supervise it before you commit:

| Horizon | Train positives | Test positives (of 27,935 at risk) |
|---|---:|---:|
| 30 days | 545 | 532 |
| 90 days | 1,361 | 1,472 |

Five hundred positives is a training set. The 90-day label has more, but it answers "who leaves this quarter" — a slower, less urgent question than the one Priya asked. Ship the 30-day horizon, and write it into `metrics.json` (`horizon_days`) and your write-up. [Week 8](week-08.md) is the lesson; this is the decision.

## 3 — Train, then prove it beats nothing

`train()` backtests: it learns on the snapshot 30 days before `as_of` and tests on the `as_of` snapshot ([Week 15](week-15.md) explains why not a `signup_date` cut). `gate(candidate, None)` refuses anything that loses to the dummy. On this file the candidate scores ROC-AUC ≈ 0.75 and PR-AUC ≈ 0.05 against a dummy's 0.019 — under three times better than guessing, on an event that happens to one customer in fifty each month. That is an ordinary churn model, not a bad one.

## 4 — The threshold is a headcount

Priya can call 80. The threshold is whatever score the 80th name has; it falls out of the list. `select` + `threshold_for` do it, and the capstone writes the result into `metrics.json`.

Then read `flag_rate` in that file. It says about 8% of customers clear the threshold — some 2,200 people, not 80. Look at `ties_at_threshold`: roughly 2,175 customers share the 80th score *exactly*. They are brand-new free accounts that signed up, logged one event, and never came back. The trees see no difference between them, so they all get the same score, and the 80th slot lands in the middle of that plateau.

!!! warning "Watch out — ties at the cut"

    A threshold only reproduces a list when scores are distinct around the cut. When thousands tie, `score >= threshold` flags thousands, and *which* 80 of them make the list is decided by sort order — unstable sorts give a different list, and a different precision, on every run. Rank and cut with an explicit tie-break (`select` and `score_batch` sort by score, then `user_id`), and treat `flag_for_cs` as "at least this risky," not "on the list."

### What the list is worth

Be honest about this one. With 532 churners among 27,935 at-risk customers, a *random* 80 reaches about 1.5 of them. The model's 80 reach about twelve — an eight-fold lift. `metrics.json` also carries `precision_at_80_ci95`: roughly 6% to 24%. On a given Monday that is anywhere from five to nineteen real churners, and [Week 11](week-11.md) showed the list is not reliably better than "sort by low usage" on a single week.

!!! math "Math, translated"

    **Lift** = precision of your list ÷ base rate. `0.15 / 0.019 ≈ 8`. Lift says the model is working. It does not say the list is worth 80 calls. That depends on what a call costs, what a saved customer is worth, and whether a call saves anyone at all — which only a randomized holdout can tell you ([Week 11](week-11.md)).

Whether to ship the desk list at that rate is a business call. The model is the same either way; the *brief* is what changes — the [scenario bank](capstone-scenarios.md) judges the same scores four other ways.

## 5 — The contract, tested

One real `predict()` call returning exactly `{churn_score, flag_for_cs, model_version}`, and three payloads `validate()` must reject: an extra key, a NaN, a missing feature. [Week 15](week-15.md) wrote `validate`; the capstone makes you prove it holds against *your* artifact.

## 6 — Promote, score, schedule

`promote()` runs the gate before it writes `artifacts/prod` — and the gate re-scores prod on the candidate's own holdout, so the two models are compared on the same customers. `score_batch(score_date, prod, limit=80)` writes the list, and refuses a score date earlier than the model's `labels_known_by`. The cron is one line of [`pipelines.job`](week-16.md): train as of today − 30 days, gate, promote, score today. After this step, nobody runs anything by hand.

!!! warning "Watch out — a job that trains and scores on the same date only works looking back"

    `train("2024-06-01")` is graded on churn between June 1 and July 1. Run that on the morning of June 1 and the labels don't exist yet — the backtest is reading next month. On a live Monday, train as of *Monday − horizon* (the latest snapshot whose outcomes you've seen), then score Monday. `metrics.json` records `labels_known_by`; `score_batch` checks it.

## 7 — The incident

Two weeks later, Priya: *half of Monday's list are people I've never heard of.* The model file hasn't changed. `validate()` passed every row. The gate passed. Something upstream did.

`incident_frame(night, seed)` is the frame the job scored that night — honest features with one seeded pipeline defect. Your seed picks which. You find it the [Week 17](week-17.md) way: row counts, then a column summary against last week, then a slice, then one customer.

```
 column           last week   tonight    ← normal week-over-week movement is ~1%
 mean ratio          1.00      ≫ or ≪ 1   a unit change, a fan-out
 zero share          0.09      jumps      an extract that stopped landing — maybe only for some customers
```

`diagnose(seed, columns)` is `True` only when you name exactly the corrupted columns. The grade is the diagnosis, not a metric.

!!! warning "Watch out — `validate()` is a contract, not a monitor"

    Every defect in step 7 has the right keys, the right types, and no NaNs. A contract rejects *malformed* input; it cannot tell you a dollar column is now in cents or that a join doubled a count. That takes a check on the *distribution* — a range, a ratio to last week, a zero share per slice — running before `score_batch`, not after Priya notices.

## Ship / don't ship

!!! success "Ship / don't ship"

    **Ship** when the candidate beats the dummy on a backtest, the threshold is a headcount someone staffed, the contract is tested against the artifact, the job runs without you, and there is a distribution check between the extract and the scorer. **Don't ship** a list whose value you can't state in the stakeholder's units — or a job whose only guard is a type check.

## ✍️ Exercise

[Capstone exercises](exercises/capstone-ship.md) — seven steps in `exercises/ml/capstone-ship/starter.py`, then the postmortem. `pytest tests/test_capstone_ship.py` runs the reference end to end.

## 🤔 Reflection

1. Your list reaches about twelve churners in 80 calls, where a random 80 would reach one or two. What would you need to know to tell Helen whether that's worth a CS salary?
2. Which step-7 defect would a range check on each column catch, and which needs a check on a *slice*?
3. You chose the 30-day horizon over the 90-day one with more positives. Where in the artifact does that choice stay visible after you've moved on — and what breaks if next quarter's retrain quietly uses 90?
