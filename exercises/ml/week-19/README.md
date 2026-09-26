# Exercise — Week 19 — RNNs: A Clipboard That Walks the Sequence

## What you are building

A mean-pooled hidden state, a reversed-week run, and a one-sentence forget-gate explanation.

## Predict before you run

1. Does `out.mean(dim=1)` give a mid-sequence dip more say than the last step?
2. If `torch.flip` drops test AUC, was the model using order or the total?
3. What does a forget gate throw away, in shopping-cart language?

## Before you start

- CPU is enough; this is a 12-step sequence toy, not a language model.
- `load_weekly_usage_grid` returns the 12 weeks before 2024-06-01 and the 30-day label after it (~2% positive). Judge models on PR-AUC or AUC; accuracy just echoes the base rate.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-19/starter.py
```

**1. Use the mean hidden state** instead of the last step (`out.mean(dim=1)`). Does a mid-sequence dip get more say?

<details>
<summary>Hint 1 — a nudge</summary>

`out[:, -1, :]` is the clipboard after week 12 — whatever survived eleven overwrites. `out.mean(dim=1)` gives every week's clipboard an equal vote. Which one lets week 6 speak?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Add a `pool` argument to the lesson's `SequenceNet` and switch between `out[:, -1, :]` and `out.mean(dim=1)`. Train both with the same seed. Accuracy sits near the majority baseline at ~2% churn — compare AUC too.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from lib.course_data import load_weekly_usage_grid

X, y = load_weekly_usage_grid(random_state=1)
idx = np.random.default_rng(1).permutation(len(X))
cut = int(0.8 * len(X))
Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

class SequenceNet(nn.Module):
    def __init__(self, pool: str = "last", hidden: int = 16):
        super().__init__()
        self.pool = pool
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        summary = out[:, -1, :] if self.pool == "last" else out.mean(dim=1)
        return self.head(summary).squeeze(-1)

def fit_eval(pool: str, train_x=Xtr, test_x=Xte, epochs: int = 30) -> tuple[float, float]:
    torch.manual_seed(0)
    model = SequenceNet(pool)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb, yb = torch.tensor(train_x.copy()), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(xb), yb).backward()
        opt.step()
    with torch.no_grad():
        p = model(torch.tensor(test_x.copy())).sigmoid().numpy()
    return float(((p > 0.5) == yte).mean()), float(roc_auc_score(yte, p))

for pool in ("last", "mean"):
    acc, auc = fit_eval(pool)
    print(f"pool={pool:<4}  acc={acc:.3f}  AUC={auc:.3f}")
```

</details>

**2. Reverse the weeks** (`torch.flip`, or `X[:, ::-1]`). Train and test on reversed sequences. What changed?

<details>
<summary>Hint 1 — a nudge</summary>

A model that only adds the weeks up gets the same total forwards and backwards. What would a model that *reads in order* notice?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Flip both train and test (`[:, ::-1]`), retrain, and compare with the forward run. If AUC drops by more than a rerun's seed spread, the model was using order. If nothing moves, it was summing.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
acc_f, auc_f = fit_eval("last")
acc_r, auc_r = fit_eval("last", train_x=Xtr[:, ::-1], test_x=Xte[:, ::-1])
print(f"forward  AUC={auc_f:.3f}   reversed AUC={auc_r:.3f}")
```

</details>

**3. One-sentence LSTM.** Explain a forget gate to a PM who has used a shopping cart.

<details>
<summary>Hint 1 — a nudge</summary>

A cart holds things across page views. What decides whether an item is still there on the next page?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Map the gate to a single cart action: at each step it decides, per item, how much to keep versus drop — learned from data, not hard-coded.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
A forget gate is the cart deciding, on every <step>, how much of each <item> to <keep/drop> — learned from <what>.
```

</details>

## Success criteria

- Mean vs last-step comparison.
- Reversed-sequence result interpreted.
- One PM sentence on the forget gate.

## After you run

An RNN is a clipboard that walks. Transformers (next week) jump instead of walking.

## Lesson link

[Week 19 — RNNs: A Clipboard That Walks the Sequence](../../../docs/ml/week-19.md)
