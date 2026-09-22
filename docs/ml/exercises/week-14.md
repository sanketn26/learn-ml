---
description: Compare a linear MLP to logistic regression, train an oversized neural net, and diagnose a PyTorch training loop that runs but does not learn.
---

# Exercises — Week 14 — Neural Nets, Without the Mystique

## What you are building

A linear MLP vs logistic regression, an oversized net, a five-line VP memo, and a training loop with `zero_grad` commented out.

## Predict before you run

1. With `activation="identity"`, will MLP AUC rhyme with logistic regression?
2. `(128, 128, 128)` on this table: train AUC vs test?
3. If you skip `opt.zero_grad()`, does loss explode, freeze, or look fine?

## Before you start

- CPU is enough. Do not reach for CUDA.
- sklearn's `MLPClassifier` is fine for tasks 1–2; task 4 needs the hand-rolled PyTorch loop from the lesson.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-14/starter.py
```

**1. Linear MLP.** Set `activation="identity"` (no ReLU). Compare AUC to logistic regression.

??? tip "Hint 1 — a nudge"
    Stack two linear functions and simplify the algebra. What kind of function comes out the other end — and which model on your list already is exactly that, plus a sigmoid?

??? tip "Hint 2 — the approach"
    The lesson's backtest split (`snapshot_split`) and `make_preprocessor()`; fit `LogisticRegression(max_iter=1000)` and `MLPClassifier(hidden_layer_sizes=(16, 8), activation="identity")` on the same transformed matrices. Compare test `roc_auc_score`. The lesson's "collapse demo" is the picture for why.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor
    from pipelines.split import snapshot_split

    train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=90)
    prep = make_preprocessor()
    X_train_t = prep.fit_transform(train[FEATURE_COLS])
    X_test_t = prep.transform(test[FEATURE_COLS])


    def auc(model) -> float:
        return roc_auc_score(y_test, model.predict_proba(X_test_t)[:, 1])


    logreg = LogisticRegression(max_iter=1000).fit(X_train_t, y_train)
    linear_mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="identity",
                               max_iter=50, random_state=42).fit(X_train_t, y_train)
    print(f"logreg AUC={auc(logreg):.3f}   identity MLP AUC={auc(linear_mlp):.3f}")
    ```

**2. Too much net.** `hidden_layer_sizes=(128, 128, 128)` on this data. What happens to train vs test?

??? tip "Hint 1 — a nudge"
    Count the weights in three layers of 128 and compare that to the number of columns in your matrix. How much room does the net have to memorize?

??? tip "Hint 2 — the approach"
    Same matrices, `MLPClassifier(hidden_layer_sizes=(128, 128, 128), activation="relu")`. Score AUC on *both* `X_train_t` and `X_test_t`. The number you care about is the gap, not either AUC alone.

??? example "Hint 3 — most of the code"
    ```python
    big = MLPClassifier(hidden_layer_sizes=(128, 128, 128), activation="relu",
                        max_iter=30, random_state=42).fit(X_train_t, y_train)
    train_auc = roc_auc_score(y_train, big.predict_proba(X_train_t)[:, 1])
    print(f"big net  train AUC={train_auc:.3f}  test AUC={auc(big):.3f}")
    ```

**3. Decision memo.** Write five lines to your VP: why CloudWave's churn model will stay a GBT this quarter.

??? tip "Hint 1 — a nudge"
    A VP reads outcomes and costs, not architectures. What did the net buy you, and what would it cost to run and explain?

??? tip "Hint 2 — the approach"
    Use your own numbers from tasks 1–2 and the lesson's GBT row. One line each: the result, the comparison, the cost, the risk, and what evidence would change your mind.

??? example "Hint 3 — a skeleton"
    ```text
    1. Result:   on the backtest holdout, GBT AUC <x> vs best net <y>.
    2. Why:      <what kind of data this is, and what trees are good at>.
    3. Cost:     <training time / tuning / explainability difference>.
    4. Risk:     <what the net did in task 2>.
    5. Revisit:  <the data or signal that would make a net worth it>.
    ```

**4. Break the loop.** Comment out `opt.zero_grad()` and rerun 5 epochs. What happens to the loss?

??? tip "Hint 1 — a nudge"
    `loss.backward()` doesn't *set* `.grad` — it *adds to* it. What does `.grad` hold on epoch 5 if nobody ever clears it?

??? tip "Hint 2 — the approach"
    Copy the lesson's four-call loop into a function with a `zero_grad: bool` flag, reseed, and run it both ways for a few epochs. Print the loss per epoch side by side. Accumulated gradients are a running sum of old directions, not "momentum."

??? example "Hint 3 — most of the code"
    ```python
    import torch
    import torch.nn as nn

    Xt = torch.tensor(np.asarray(X_train_t, dtype=np.float32))
    yt = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)


    def run(zero_grad: bool, epochs: int = 5) -> list[float]:
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(Xt.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        losses = []
        for _ in range(epochs):
            if zero_grad:
                opt.zero_grad()
            loss = loss_fn(net(Xt), yt)
            loss.backward()
            opt.step()
            losses.append(round(float(loss), 4))
        return losses


    print("with zero_grad   ", run(True))
    print("without zero_grad", run(False))
    ```
    Reading the two rows is yours.

## Success criteria

- Identity MLP vs logreg AUCs.
- Deep-net train/test gap.
- Five-line memo.
- Loss behavior without `zero_grad`.

## After you run

A net is mixers + switches + a four-line training step. On this SaaS table the GBT still ships.

## Lesson link

[Week 14 — Neural Nets, Without the Mystique](../week-14.md)
