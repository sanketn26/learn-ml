# Exercise — Week 16 — The Job Pipeline

Ana's rule: nothing writes to `artifacts/prod` unless it beats the dummy and beats current prod, automatically, every night — no exceptions for "it looked fine on my laptop." Wire the gate she'll actually trust, then prove it refuses a losing candidate.

## What you are building

A candidate directory, a promote gate, proof that train does not write prod, and a five-line cron.

## Predict before you run

1. After two trains with the same `--as-of`, did `artifacts/prod` change?
2. If you raise dummy PR-AUC above the candidate, does promote refuse?
3. Who is allowed to write `artifacts/prod`?

## Before you start

- `train` backtests on the 30-day horizon (~545 training positives), writes the horizon and a bootstrap interval on precision@80 into `metrics.json`, and the gate refuses to compare against a prod model trained on a different horizon.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
pytest tests/
python exercises/ml/week-16/starter.py
```

Full pipeline (after you trust the starter):

```bash
python -m pipelines.train --as-of 2024-06-01
python -m pipelines.promote --candidate artifacts/20240601
python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv
head tonight.csv
```

**1. Gate.** After training, open `artifacts/20240601/metrics.json`. Confirm `pr_auc > dummy_pr_auc`. If you temporarily set the dummy higher in a scratch copy of `promote.gate`, the promote must refuse.

<details>
<summary>Hint 1 — a nudge</summary>

The gate is a pure function of `metrics.json`. To test that it says *no*, you don't need a worse model — you need a worse *file*.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Call `pipelines.train.train(...)` into a directory you control, read its `metrics.json`, then copy the candidate to a scratch dir, raise `dummy_pr_auc` above `pr_auc` in the copy, and call `pipelines.promote.gate(scratch, None)`. It returns `(ok, reason)`.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import json
import shutil
from pathlib import Path

from pipelines.promote import gate
from pipelines.train import train

meta = train("2024-06-01", Path("artifacts"))
candidate = Path("artifacts") / meta["model_version"]
print("pr_auc", meta["pr_auc"], "dummy", meta["dummy_pr_auc"], "→", gate(candidate, None))

scratch = Path("artifacts") / "scratch-worse"
shutil.copytree(candidate, scratch, dirs_exist_ok=True)
worse = json.loads((scratch / "metrics.json").read_text())
worse["dummy_pr_auc"] = worse["pr_auc"] + 0.05
(scratch / "metrics.json").write_text(json.dumps(worse))
print("rigged dummy →", gate(scratch, None))
```

</details>

**2. Train does not write prod.** Run train twice with the same `--as-of`. `artifacts/prod` must change only after `promote`.

<details>
<summary>Hint 1 — a nudge</summary>

How would you *prove* a directory didn't change? Not by looking at it — by fingerprinting it before and after.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Hash (or just record the `mtime` of) `artifacts/prod/metrics.json` if it exists, run `train` twice, and compare. Then `promote` and compare again. Only the last comparison should differ.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import hashlib

from pipelines.promote import promote

def fingerprint(d: Path) -> str:
    f = d / "metrics.json"
    return hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.exists() else "missing"

prod = Path("artifacts") / "prod"
before = fingerprint(prod)
for _ in range(2):
    train("2024-06-01", Path("artifacts"))
print("after two trains:", before, "→", fingerprint(prod))
try:
    promote(candidate, prod)
except SystemExit as refused:  # a refusal is the gate working, not a crash
    print(refused)
print("after promote:   ", fingerprint(prod))
```

</details>

**3. One function.** In `tests/test_features.py`, add (or just read) the assertion that `FEATURE_COLS` never intersects `FORBIDDEN`.

<details>
<summary>Hint 1 — a nudge</summary>

If training and scoring each built features their own way, you'd have two products. What single function do both of them import?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Search `tests/test_features.py` for `FORBIDDEN` first. If the check isn't there, it's a one-line set intersection in a new test function.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
from pipelines.features import FEATURE_COLS, FORBIDDEN

assert not set(FEATURE_COLS) & set(FORBIDDEN), set(FEATURE_COLS) & set(FORBIDDEN)
```

</details>

**4. Cron.** Write a five-line shell script you would hang on a weekly timer: pytest, train, promote, score. Do not add Airflow.

<details>
<summary>Hint 1 — a nudge</summary>

Four commands, and each one should stop the next from running if it fails. What shell option gives you that for free?

</details>

<details>
<summary>Hint 2 — the approach</summary>

`set -euo pipefail` on line one, then the four commands from the top of this page in order. A refused promote exits non-zero — that should stop the score step from running on a stale model? Decide, and write the answer as a comment.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```bash
set -euo pipefail
python -m pytest tests/
python -m pipelines.train --as-of "<date>"
python -m pipelines.promote --candidate "artifacts/<version>"
python -m pipelines.score_batch --as-of "<date>" --artifact artifacts/prod --out tonight.csv
```

</details>

## Success criteria

- metrics.json beats dummy.
- Promote refuses a worse dummy.
- Prod unchanged across two trains.
- Five-line cron, no Airflow.

## After you run

The job is train → gate → prod dir → tonight's CSV. Kubeflow is not the week. You're on-call starting next week — this gate is what you're trusting at 3 a.m.

## Lesson link

[Week 16 — The Job Pipeline](../../../docs/ml/week-16.md)
