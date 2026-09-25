---
description: Tune a 1-D convolutional neural network's kernel size, benchmark it next to a dense-flatten model, and sketch how its sliding stencil scans a sequence.
---

# Exercises — Week 18 — CNNs: Sliding Detectors

## What you are building

A 1-D CNN with a wider kernel, a dense-flatten baseline, and a stencil sketch on 12 week-boxes.

## Predict before you run

1. Does `kernel_size=5` move test PR-AUC, or just change what the detector looks at?
2. If a `Linear(12, 1)` ties the CNN, was the signal *shape* or *total*?
3. Which of three stencil positions fires on a late-week drop?

## Before you start

- This is a teaching toy on 12 weekly totals. CPU only.
- `load_weekly_usage_grid` returns the 12 weeks before 2024-06-01 and the 30-day label after it (~2% positive). Judge models on PR-AUC or AUC; accuracy just echoes the base rate. Don't ship anything from it.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-18/starter.py
```

**1. Kernel size.** Change `kernel_size` to 5. Does test PR-AUC move? What did you make the detector look at?

??? tip "Hint 1 — a nudge"
    A kernel of 3 sees three consecutive weeks at a time. What pattern could a 5-week window catch that a 3-week one can't — and does this data even have it?

??? tip "Hint 2 — the approach"
    Give `UsageCNN` a `kernel_size` argument and train both versions with the same seed and split. With ~2% churners, accuracy hugs the majority baseline — print test AUC next to it so a real change is visible.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    from lib.course_data import load_weekly_usage_grid

    X, y = load_weekly_usage_grid()
    idx = np.random.default_rng(0).permutation(len(X))
    cut = int(0.8 * len(X))
    Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


    class UsageCNN(nn.Module):
        def __init__(self, kernel_size: int = 3):
            super().__init__()
            self.conv = nn.Conv1d(1, 8, kernel_size=kernel_size)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Linear(8, 1)

        def forward(self, x):
            h = F.relu(self.conv(x.unsqueeze(1)))
            return self.head(self.pool(h).squeeze(-1)).squeeze(-1)


    def fit_eval(model: nn.Module, epochs: int = 12) -> tuple[float, float]:
        torch.manual_seed(0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        xb, yb = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(xb), yb).backward()
            opt.step()
        with torch.no_grad():
            p = model(torch.tensor(Xte)).sigmoid().numpy()
        return float(average_precision_score(yte, p)), float(roc_auc_score(yte, p))


    for k in (3, 5):
        torch.manual_seed(0)
        pr, auc = fit_eval(UsageCNN(kernel_size=k))
        print(f"kernel={k}  test PR-AUC={pr:.3f}  AUC={auc:.3f}  (base rate {yte.mean():.3f})")
    ```

**2. Dense baseline.** Flatten the 12 weeks into a `nn.Linear(12, 1)` and compare.

??? tip "Hint 1 — a nudge"
    A single linear layer over 12 numbers can learn "total usage" and "recent vs early" — but not "a spike anywhere." If it keeps up with the CNN, what does that tell you the CNN was actually using?

??? tip "Hint 2 — the approach"
    `nn.Linear(12, 1)` is logistic regression on the 12 weekly values. Train it with the same `fit_eval`. A tie means the *shape* wasn't the signal — stop claiming the CNN "saw the drop-off."

??? example "Hint 3 — most of the code"
    ```python
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(12, 1)

        def forward(self, x):
            return self.lin(x).squeeze(-1)


    torch.manual_seed(0)
    pr, auc = fit_eval(Dense())
    print(f"dense      test PR-AUC={pr:.3f}  AUC={auc:.3f}")
    ```
    Whether the CNN earned its extra weights is your call.

**3. Draw it.** Sketch one user as 12 boxes and a 3-wide stencil in three positions. Circle the position you think fires on a late-week drop.

??? tip "Hint 1 — a nudge"
    A stencil fires when the three boxes under it look like the pattern it learned. What do three boxes look like when a drop is inside them?

??? tip "Hint 2 — the approach"
    Pick a real row from `Xte` whose last weeks fall off, draw its values as boxes, and slide `[ ][ ][ ]` over weeks 1–3, 6–8, and 10–12. Note what each position sees.

??? example "Hint 3 — a skeleton"
    ```text
    week:   1  2  3  4  5  6  7  8  9 10 11 12
    usage: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
            └─A─┘          └─B─┘       └─C─┘
    A sees: <…>   B sees: <…>   C sees: <…>   → fires on the drop: <A/B/C>, because <…>
    ```

## Success criteria

- Kernel-5 vs kernel-3 note.
- Dense baseline PR-AUC and AUC next to the CNN.
- ASCII or paper sketch with a circled position.

## After you run

A convolution is one detector, many places. On CloudWave's 7-column table, last week's GBT still ships.

## Lesson link

[Week 18 — CNNs: Sliding Detectors](../week-18.md)
