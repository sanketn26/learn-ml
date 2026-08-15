# Week 14 — RNNs: A Clipboard That Walks the Sequence

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a fold / reduce, a state machine, or a running total.

An **RNN** (Recurrent Neural Network) is a loop with memory: *read the next token, update a hidden state, repeat.*

---

## 🎯 What you will be able to do

- Draw an RNN as “same function, new input, old clipboard”
- See why long memory fades (vanishing gradients — in engineer English)
- Know what LSTM / GRU add (gates = locks on the clipboard)
- Train a tiny RNN on CloudWave weekly usage
- Know when to skip RNNs and go to Transformers (next week)

!!! think "Think of it like… a running total, or a state machine."

    You walk a user’s weeks from oldest to newest. After each week you update a small vector — the **hidden state** — the way a cashier updates the subtotal. At the end, that vector is “everything I still remember about this customer.” A linear layer turns it into a churn score.

## If you already write software

An RNN is a `reduce` with a clipboard.

```python
state = empty
for token in sequence:
    state = f(state, token)    # same function, new input, old clipboard
return state                   # or emit something every step
```

That is a recurrent net. The “recurrent” part is: **one function, reused, carrying a hidden state forward.**

```
for-loop over events          RNN
fold / reduce                 hidden state
iterator                      the time axis
a bug where you forget        vanishing gradient (memory fades)
```

### Why memory fades — and what LSTM/GRU add

A vanilla RNN multiplies the clipboard by a matrix at every step. Multiply a number by 0.9 twelve times and it is basically gone. Early weeks of usage never reach the churn decision. That is vanishing gradient, in one sentence.

LSTM / GRU add **locks** on the clipboard: forget gate, input gate, output gate. They are not a new philosophy. They are valves so the clipboard can *keep* a fact (“this user was a whale in week 1”) for a long time.

### When this is the wrong tool

If you only have a single row per user, there is no sequence. If the sequence is short and positional (12 weeks of usage), a 1-D CNN or a tiny Transformer may be simpler. RNNs still matter as a mental model — “state + new input → new state” — and they still show up in streaming systems.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all 50k rows.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the shared style kit importable from the repo root

from pathlib import Path
import sys
from lib.course_data import find_data_dir

DATA = find_data_dir()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Install with:  pip install torch") from exc

torch.manual_seed(0)
DEVICE = torch.device("cpu")
print("torch", torch.__version__, "device", DEVICE)
```

## Unrolling the loop

```
h0 = zeros
h1 = f(h0, week1)
h2 = f(h1, week2)
...
hT = f(hT-1, weekT)  →  score

f is THE SAME function every step. That is the whole trick.
```

!!! math "Math, translated — vanishing memory"

    Training walks backward through those T steps. If each step multiplies the “how much should I remember?” signal by a number < 1, after 30 steps the early weeks have been multiplied into dust. The model becomes “whatever happened last.” **LSTMs / GRUs** add gates: learned locks that can say “keep this bit unchanged.” They do not magically remember 10,000 steps. They just forget slower.

```python
# Memory fade cartoon: multiply a signal by 0.7, 30 times
steps = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(8, 3.2))
for rho, label in [(0.95, "gate mostly open (0.95)"),
                   (0.7, "typical tanh/sigmoid (0.7)"),
                   (0.4, "closed-ish (0.4)")]:
    ax.plot(steps, rho ** steps, label=label)
ax.set_xlabel("steps back in time")
ax.set_ylabel("how much of week-1 still remains")
ax.set_title("Why vanilla RNNs forget the signup week")
ax.legend()
plt.tight_layout()
plt.show()
```

## Same CloudWave sequences as Week 13, new `forward`

```python
X, y = load_weekly_usage_grid(DATA, random_state=1)
rng = np.random.default_rng(1)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
Xtr, Xte = X[idx[:cut]], X[idx[cut:]]
ytr, yte = y[idx[:cut]], y[idx[cut:]]
print(Xtr.shape, "churn", float(ytr.mean()))
```

## Vanilla RNN, then a GRU

`nn.RNN` / `nn.GRU` want `(batch, time, features)` when `batch_first=True`.

```python
class SequenceNet(nn.Module):
    def __init__(self, kind="gru", hidden=16):
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU}[kind]
        self.rnn = cell(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T) → (B, T, 1)
        out, h = self.rnn(x.unsqueeze(-1))
        last = out[:, -1, :]                 # clipboard after the final week
        return self.head(last).squeeze(-1)

def fit(kind, epochs=8):
    model = SequenceNet(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    rows = []
    for _ in range(epochs):
        model.train()
        xb = torch.tensor(Xtr)
        yb = torch.tensor(ytr, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            te = F.binary_cross_entropy_with_logits(
                model(torch.tensor(Xte)), torch.tensor(yte, dtype=torch.float32)
            )
        rows.append((float(loss), float(te)))
    return model, np.array(rows)

fig, ax = plt.subplots(figsize=(8, 3.6))
for kind, color in [("rnn", "#64748b"), ("gru", "#1d4ed8")]:
    _, hist = fit(kind)
    ax.plot(hist[:, 0], ls="--", color=color, alpha=0.5)
    ax.plot(hist[:, 1], color=color, label=f"{kind} test loss")
ax.set_title("Dashed = train, solid = test — GRU should forget slower")
ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout()
plt.show()
```

## LSTM / GRU gates, in English

| Gate | Plain English |
|---|---|
| Forget / reset | “Throw this bit of the clipboard away” |
| Input / update | “Write the new week in” |
| Output | “What part of the clipboard is the answer *this* step” |

You do not tune gates by hand. The training loop learns when to lock.

!!! engineer "Engineer mental model"

    An RNN is a `for` loop you can backprop through. That loop is **sequential** — week 7 cannot start until week 6 is done — so GPUs hate long RNNs. Transformers (next week) look at every week at once. That is why industry moved.


!!! warning "Watch out"

    Teacher forcing, packed sequences, bidirectional RNNs — you will see the words. They are implementation details around the same clipboard. Do not start there. Start with “last hidden state → linear.”


!!! success "Ship / don’t ship"

    A small GRU is still fine for *short* sensor traces and on-device models. For language, search, and anything longer than a few dozen steps, ship a Transformer (or call an API that already is one).


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-14.md). Starter: `python exercises/ml/week-14/starter.py` from the repo root.

## 🤔 Reflection

1. How is an RNN like a fold/reduce? How is it not (the body is learned)?
2. Why did vanishing gradients hurt signup-week features more than last-week features?
3. Why are RNNs slow on a GPU compared to a CNN or a Transformer?

## 🔗 Next week

Transformers: throw away the clipboard. Every week (or word) looks at every other one, in parallel.
