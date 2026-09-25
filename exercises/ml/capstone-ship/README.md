# Exercise — Capstone — Ship the Churn Score

Every step below has a function in `pipelines/` already. The capstone is the composition — and two decisions the weeks never made for you: how many names the list holds, and what broke on the night it went wrong.

## What you are building

A candidate model that beats the dummy on a backtest, a threshold set by Priya's capacity, a contract test, a promoted artifact that scores tonight's list, and a postmortem for an incident you did not cause.

## Predict before you run

1. With a 30-day horizon (~2% base rate), roughly how many of Priya's 80 calls will reach a customer who actually leaves — and how many would a random 80?
2. Will a 0.5 threshold flag more or fewer than 80 customers? Will the 80th score's threshold flag *exactly* 80?
3. On the incident night, how many rows will `validate()` reject?
4. Which is easier to see in a column summary: a column that doubled, or one that went to zero for a slice of customers?

## Before you start

- Run from the repo root. The starter writes under `artifacts/capstone-ship/` (gitignored).
- Finish Weeks 15–17 first. This page does not re-teach `build_features`, `snapshot_split`, `train`, `promote`, or `score_batch` — it composes them.
- `capstone_ship/incident.py` is the answer key for step 7. Don't read it before you've written the postmortem.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/capstone-ship/starter.py
pytest tests/test_capstone_ship.py
```

**1. As-of features.** Implement `step1_features`: the at-risk frame at `AS_OF`, with asserts for grain and time.

<details>
<summary>Hint 1 — a nudge</summary>

Week 3's two questions, as asserts: one row per what? And could any row know about the future?

</details>

<details>
<summary>Hint 2 — the approach</summary>

`build_features(as_of, n=None)`. Assert `user_id` is unique and no `signup_date` is after `as_of`. Add one assert that the lifetime columns (`tenure_days`, `churn_date`) are not in `FEATURE_COLS`.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone_ship.briefs import RETENTION_DESK, judge, select, threshold_for
from capstone_ship.incident import column_shift, diagnose, incident_frame
from pipelines.contract import load_artifact, predict, validate
from pipelines.features import FEATURE_COLS, NUMERIC, build_features
from pipelines.promote import gate, promote
from pipelines.score_batch import _payload, score_batch
from pipelines.split import snapshot_split
from pipelines.train import train

AS_OF, HORIZON = pd.Timestamp("2024-06-01"), 30
WORKDIR = Path("artifacts") / "capstone-ship"

frame = build_features(as_of=AS_OF, n=None)
assert frame["user_id"].is_unique
assert frame["signup_date"].max() <= AS_OF
print(f"{len(frame):,} at-risk customers")
```

</details>

**2. A label with a horizon.** Implement `step2_labels` with an explicit horizon. In your write-up, say why that horizon and not 90 days.

<details>
<summary>Hint 1 — a nudge</summary>

Priya's real question is "who leaves in the next month?" How many of those events does this file actually contain — and what does Week 8 say about supervising a label that rare?

</details>

<details>
<summary>Hint 2 — the approach</summary>

`snapshot_split(AS_OF, horizon_days=h)` for `h` in 30 and 90. Print positives on each side. Pick the shortest horizon that leaves both sides with enough positives to learn from, and write the trade-off down — a longer horizon is an easier, less urgent question.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
for h in (30, 90):
    _, y_tr, _, y_te = snapshot_split(AS_OF, horizon_days=h)
    print(f"horizon={h:>3}d  train positives={int(y_tr.sum()):>4}  test positives={int(y_te.sum()):>4}")
train_df, y_train, test_df, y_test = snapshot_split(AS_OF, horizon_days=HORIZON)
```

</details>

**3. Train and beat the dummy.** Implement `step3_train`: a candidate artifact that passes the gate on the backtest.

<details>
<summary>Hint 1 — a nudge</summary>

You already wrote this job in Week 16. What's the one function call, and what decides whether its output is allowed anywhere near prod?

</details>

<details>
<summary>Hint 2 — the approach</summary>

`train(as_of, out_dir, horizon_days=...)` writes `out_dir/<version>/`. Then `gate(candidate, None)` returns `(ok, reason)`. Assert `ok` — a capstone that ships a model losing to the dummy has failed step 3, not passed it.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
out = WORKDIR / "artifacts"
meta = train(str(AS_OF.date()), out, horizon_days=HORIZON)
candidate = out / meta["model_version"]
ok, reason = gate(candidate, None)
print({k: meta[k] for k in ("auc", "pr_auc", "dummy_pr_auc", "precision_at_80", "base_rate")}, ok, reason)
```

</details>

**4. Threshold from capacity.** Implement `step4_threshold`: use the brief's capacity to pick the list and its threshold, judge it on the backtest labels, and write the threshold into `metrics.json`.

<details>
<summary>Hint 1 — a nudge</summary>

Priya's budget is a *count*. The threshold is whatever score the 80th name happens to have — it falls out of the list, not the other way round.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Score `test_df` with the candidate's pipeline. `select(test_df, scores, RETENTION_DESK)` gives the list, `threshold_for(picked)` its cut, `judge(...)` its hits against `y_test`. Count what 0.5 would have flagged. Update `metrics.json` with the threshold, the brief key, and the capacity so `predict()` flags the same people the list contains.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
art = load_artifact(candidate)
scores = art["pipeline"].predict_proba(test_df[FEATURE_COLS])[:, 1]
picked = select(test_df, scores, RETENTION_DESK)
verdict = judge(picked, test_df, y_test, RETENTION_DESK)
cut = threshold_for(picked)
print(verdict, f"threshold={cut:.4f}", f"flagged at 0.5={(scores >= 0.5).sum()}")

metrics = json.loads((candidate / "metrics.json").read_text())
metrics.update({"threshold": round(cut, 4), "brief": RETENTION_DESK.key, "capacity": RETENTION_DESK.capacity})
(candidate / "metrics.json").write_text(json.dumps(metrics, indent=2))
```
Whether about a dozen real churners in 80 calls — and the bootstrap interval around that dozen — is worth Priya's week goes in the write-up.

</details>

**4b. Read the misses.** Before anything ships, pull two lists from the backtest: the 10 highest-scored customers who did **not** churn (false alarms at the top of Priya's list), and the 10 churners the model ranked lowest (misses it will never call). Print their feature rows next to the population median. In the write-up, one sentence each: what do the false alarms have in common, and what do the misses have in common that the features cannot see?

<details>
<summary>Hint 1 — a nudge</summary>

Metrics tell you *how often* the model is wrong. These two lists tell you *how* — and the "how" is usually a missing feature or a data bug, not a missing layer.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Put `scores` and `y_test` on `test_df`. Sort by score: the top rows with `y == 0` are false alarms; the bottom rows with `y == 1` are misses. Compare their `FEATURE_COLS` to `test_df[FEATURE_COLS].median()`. Week 15's reason codes help explain a single row.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
ranked = test_df.assign(score=scores, y=y_test.to_numpy()).sort_values(["score", "user_id"], ascending=[False, True])
false_alarms = ranked[ranked["y"] == 0].head(10)
misses = ranked[ranked["y"] == 1].tail(10)
cols = ["plan_type", "mrr", "tenure_so_far", "log_usage", "total_events", "n_support"]
print("population median:\n", test_df[cols[1:]].median().round(2).to_string())
print("\ntop false alarms:\n", false_alarms[cols + ["score"]].round(3).to_string(index=False))
print("\nlowest-ranked churners:\n", misses[cols + ["score"]].round(3).to_string(index=False))
```
Expect the false alarms to be accounts about two weeks old with one event: the model cannot tell *dormant* from *hasn't started yet*. Expect the misses to look like *healthy* customers — busy, paying, long-tenured — who left anyway. Nothing in `FEATURE_COLS` saw it coming. Name the column you would add (recent activity trend? a support ticket's sentiment?) and how you would compute it as of Monday.

</details>

**5. Contract test.** Implement `step5_contract`: one real `predict()` call, and three payloads `validate()` must reject.

<details>
<summary>Hint 1 — a nudge</summary>

Week 15's contract has three ways to be wrong: a key that isn't a feature, a value that's missing, a feature that isn't there. One bad payload each.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Build a real payload from one test row with `pipelines.score_batch._payload`. Assert `predict` returns exactly `churn_score`, `flag_for_cs`, `model_version`. Then feed `validate` a payload plus `user_id`, one with a NaN `mrr`, and one missing `plan_type`; each must raise.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
payload = _payload(test_df[FEATURE_COLS].iloc[0].to_dict())
response = predict(payload, load_artifact(candidate))
assert set(response) == {"churn_score", "flag_for_cs", "model_version"}

bad_payloads = {
    "extra key": {**payload, "user_id": "user_041906"},
    # "missing value": ...
    # "missing feature": ...
}
for why, bad in bad_payloads.items():
    try:
        validate(bad)
    except (ValueError, TypeError) as exc:
        print(f"rejected ({why}): {exc}")
    else:
        raise AssertionError(f"validate accepted: {why}")
```

</details>

**6. Promote and score tonight.** Implement `step6_promote_and_score`, then write the five-line cron that runs steps 3–6 every week.

<details>
<summary>Hint 1 — a nudge</summary>

Only one function may write `artifacts/prod`. And the nightly list should be as long as the desk can call — no longer.

</details>

<details>
<summary>Hint 2 — the approach</summary>

`promote(candidate, prod)` runs the gate itself. `score_batch(as_of, prod, limit=brief.capacity)` returns tonight's ranked list; save it as `tonight.csv`. The cron is Week 16's, with your horizon and capacity flags.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
prod = WORKDIR / "artifacts" / "prod"
promote(candidate, prod)
tonight = score_batch(str(AS_OF.date()), prod, limit=RETENTION_DESK.capacity)
tonight.to_csv(WORKDIR / "tonight.csv", index=False)
print(tonight.head())
```
```bash
set -euo pipefail
python -m pytest tests/
python -m pipelines.train --as-of "$AS_OF"
# promote, then score with --limit 80
```

</details>

**7. The incident.** Two weeks later, Priya says Monday's list "looks like strangers." Implement `step7_incident`: compare the incident night's frame (`incident_frame(INCIDENT_NIGHT, seed)`) with the week before, name the corrupted columns, and check them with `diagnose(seed, columns)`.

<details>
<summary>Hint 1 — a nudge</summary>

The model didn't change — `artifacts/prod` is the same file. So what else could make the same model rank different people? Start where Week 17 did: row counts, then columns, then one customer.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Build last week's clean frame (`build_features(night - 7 days)`) as the reference. `column_shift(reference, tonight, NUMERIC)` gives mean ratio and zero share per column — normal week-over-week movement is about 1%. Look for a ratio far from 1 *or* a zero share that jumped. If a zero share moved only a little overall, slice by `tenure_so_far`: a defect can hide in a subset. Count how many of the top 80 changed — that's the blast radius.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
SEED = 0
night = AS_OF + pd.Timedelta(days=14)
reference = build_features(night - pd.Timedelta(days=7), n=None)
tonight_frame = incident_frame(night, seed=SEED)
print(f"rows: last week={len(reference):,}  tonight={len(tonight_frame):,}")
print(column_shift(reference, tonight_frame, NUMERIC).round(3))

def top80(frame: pd.DataFrame) -> set:
    s = load_artifact(prod)["pipeline"].predict_proba(frame[FEATURE_COLS])[:, 1]
    return set(frame["user_id"].to_numpy()[np.argsort(-s)[:80]])

clean_tonight = build_features(night, n=None)
print("names changed on Priya's list:", 80 - len(top80(clean_tonight) & top80(tonight_frame)))
```
Which columns you name — and `diagnose(SEED, ...)` — is yours.

</details>

**8. Postmortem.** Half a page: symptom, detection, root cause, blast radius, the test that would have caught it, and your ship / don't-ship call on the list itself.

<details>
<summary>Hint 1 — a nudge</summary>

Write it for Ana, who will be on call next time. Every claim should point at a number you printed in step 7.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Week 17's format. "The test that would have caught it" should be something you could add to `tests/` today — a range check, a zero-share check by tenure bucket, a row-count ratio — and it should run *before* `score_batch`.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
Incident — <date>, seed <n>
Symptom:       <what Priya saw>
Detection:     <the table / slice that showed it; the numbers>
Root cause:    <which columns, what upstream change, why validate() passed>
Blast radius:  <names changed of 80; nights affected>
Missing test:  tests/<file>.py::<name> — asserts <what>, runs before score_batch
The list:      at <precision> vs base rate <r>, I would <ship / not ship> the desk list because <…>
```

</details>

**9. Re-skin it.** Run two briefs from the [scenario bank](../../../docs/ml/capstone-scenarios.md) through your step 4 — at least one of `discount-targeting` or `onboarding-activation` — and write a ship / don't-ship for each in its stakeholder's units.

<details>
<summary>Hint 1 — a nudge</summary>

Your step 4 already takes a `brief` argument. What *else* has to change to answer Helen instead of Priya — the model, the scores, or just the brief?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Loop over `BRIEFS` with the same `scores` and `test_df`. Read each verdict against *its own* baseline: the break-even rate against 1.0, the slice precision against the slice base rate, churners pitched against the usage-only list.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
from capstone_ship.briefs import BRIEFS

for key in ("discount-targeting", "onboarding-activation"):
    brief = BRIEFS[key]
    picked = select(test_df, scores, brief)
    print(key, f"threshold={threshold_for(picked):.4f}", judge(picked, test_df, y_test, brief))
```
The two ship / don't-ship calls are yours.

</details>

## Success criteria

- `gate(candidate, None)` is `(True, "ok")` on the backtest.
- `metrics.json` carries the threshold, brief, and capacity; tonight's list has exactly 80 names.
- Ten false alarms and ten misses read, with one sentence on what each group shares.
- Three payloads rejected by `validate()`.
- `diagnose(seed, columns)` is `True` for your seed — and your postmortem names the test that would have caught it.
- Two briefs judged in their stakeholders' units, each with a ship / don't-ship.

## After you run

`pytest tests/test_capstone_ship.py` runs the reference solution end to end for all three defects. If your step 7 works for your seed, try two more seeds: the defects are not equally loud.

## Lesson link

[Capstone — Ship the Churn Score](../../../docs/ml/capstone-ship.md)
