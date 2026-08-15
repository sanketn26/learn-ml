# Week 13 — CNNs: Sliding Detectors

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a sliding-window loop, a regex, or an image filter.

A **Convolutional Neural Network** is not “the image one.” It is: **reuse a tiny detector at every position.**

---

## 🎯 What you will be able to do

- Picture a convolution as a small stencil sliding over a sequence or an image
- Explain weight sharing (“one detector, many places”) in one sentence
- Say what pooling does (downsample, keep the loudest hit)
- Train a tiny **1-D CNN** on CloudWave usage-over-time to predict churn
- Know when a CNN is the wrong tool (most SaaS tables)

!!! think "Think of it like… Ctrl+F with a fuzzy stencil, or a antivirus signature scan."

    You do not write a separate rule for “spike on Monday” and “spike on Thursday.” You write *one* “usage spike” detector and slide it along the week. The detector’s numbers are learned. That reuse is why CNNs have so few weights compared to a giant dense net on every pixel.

## If you already write software

A convolution is a sliding-window loop you have already written.

```python
# you have done this
for i in range(len(signal) - k):
    window = signal[i : i + k]
    hits.append(dot(window, kernel))
```

A CNN **learns the kernel** and **reuses it at every position**. One “usage spike” detector, stamped along the weeks. That reuse is why a CNN has so few weights compared to a dense net that gets its own parameter per pixel.

```
Ctrl+F / regex / antivirus signature     one pattern, many places
image sharpen / blur kernel              one stencil, every pixel
1-D conv on weekly usage                 one “spike / drop-off” detector, every week
```

### When this is the wrong tool

A Customer 360 row (`mrr`, `tenure`, `plan_type`) has no spatial axis. There is nothing to slide over. Use a tree. Use a CNN when the *order* or *position* matters: usage-over-weeks, a spectrogram, an image, a token sequence (and even then, Transformers often win on text — week 15).

### Picture weight sharing

A dense layer on a 12-week series would learn a different rule for “week 1” and “week 7.” A conv layer says: a drop-off looks like a drop-off wherever it sits. That is the inductive bias. You are telling the model the world is translation-invariant, the same way a regex does not care whether the match starts at column 3 or column 30.

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

## Visual: a 1-D filter sliding

```
usage by day:   [2, 2, 8, 9, 2, 2, 3]
detector:          [−1, 2, −1]     “a spike relative to neighbors”
slide:          pos0: 2·-1 + 2·2 + 8·-1 = -6
                pos1: 2·-1 + 8·2 + 9·-1 =  5   ← hit
                pos2: 8·-1 + 9·2 + 2·-1 =  8   ← hit
```

A **2-D CNN** is the same idea on a grid (a screenshot, a heatmap, an MRI). Same stencil, two axes.

!!! engineer "Engineer mental model"

    `nn.Conv1d` is a 1-D convolution over time. `nn.Conv2d` is over height×width. `in_channels` is “how many parallel traces” (RGB = 3, or many features per day). `out_channels` is “how many different detectors we learn.” Pooling is `max` in a window — keep the strongest hit, throw away the exact timestamp.

```python
# Hand-built spike detector — no learning yet
signal = np.array([2, 2, 8, 9, 2, 2, 3], dtype=float)
kernel = np.array([-1.0, 2.0, -1.0])
hits = np.convolve(signal, kernel, mode="valid")

fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
axes[0].stem(signal)
axes[0].set_title("Daily usage")
axes[1].stem(np.arange(1, 1 + len(hits)), hits, linefmt="#dc2626", markerfmt="ro")
axes[1].axhline(0, color="#94a3b8", ls="--")
axes[1].set_title("Detector response — peaks where the stencil matches a spike")
axes[1].set_xlabel("day")
plt.tight_layout()
plt.show()
print("hits", np.round(hits, 2))
```

## CloudWave: usage as a short time series

Each user becomes a length-T vector of weekly usage. The CNN’s job: “does this *shape* look like someone about to churn?” — not “what is their total.”

```python
X, y = load_weekly_usage_grid(DATA)  # ~3k users × 12 weeks, CPU-friendly
print(f"users={len(X):,}  timesteps={X.shape[1]}  churn={y.mean():.3f}")

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
tr, te = idx[:cut], idx[cut:]
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
```

## A tiny Conv1d in PyTorch

`input` shape for Conv1d is `(batch, channels, time)`. We have one channel (usage).

```python
class UsageCNN(nn.Module):
    def __init__(self, t: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)   # loudest hit anywhere in the 12 weeks
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        # x: (B, T) → (B, 1, T)
        h = F.relu(self.conv(x.unsqueeze(1)))
        h = self.pool(h).squeeze(-1)         # (B, 8)
        return self.head(h).squeeze(-1)      # (B,) logits

def run_epoch(model, xb, yb, opt=None):
    model.train(opt is not None)
    xb = torch.tensor(xb, dtype=torch.float32)
    yb = torch.tensor(yb, dtype=torch.float32)
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, yb)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (logits.sigmoid() > 0.5).float()
        acc = float((pred == yb).float().mean())
    return float(loss), acc

model = UsageCNN(Xtr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
print(model)
print("weights:", sum(p.numel() for p in model.parameters()))

hist = []
for epoch in range(12):
    tr_loss, tr_acc = run_epoch(model, Xtr, ytr, opt)
    te_loss, te_acc = run_epoch(model, Xte, yte, opt=None)
    hist.append((tr_loss, te_loss, tr_acc, te_acc))

hist = np.array(hist)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(hist[:, 0], label="train"); axes[0].plot(hist[:, 1], label="test")
axes[0].set_title("loss"); axes[0].legend()
axes[1].plot(hist[:, 2], label="train"); axes[1].plot(hist[:, 3], label="test")
axes[1].set_title("accuracy"); axes[1].legend()
plt.tight_layout()
plt.show()
print(f"final test acc={hist[-1, 3]:.3f}  (majority baseline ~{1 - yte.mean():.3f})")
```

## 2-D picture (so “CNN” in papers makes sense)

```
image  28×28
   │  3×3 stencil, 8 detectors
   ▼
feature maps  8 × 28 × 28     “where did detector #3 fire?”
   │  max-pool 2×2
   ▼
8 × 14 × 14
   │  flatten + linear
   ▼
class scores
```

Same four-line training step as Week 11. Only the *body* of `forward` changed.

!!! warning "Watch out"

    A CNN on 12 weekly totals is a teaching toy. Real CNNs earn their keep on *grids* (product screenshots, document scans) or long dense signals. For CloudWave’s 7-column table, last week’s GBT is still the right ship.


!!! success "Ship / don’t ship"

    Ship a CNN when nearby positions *mean the same thing* (pixels, audio samples, equally spaced sensors). Do not ship one because a blog said “deep learning.”


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-13.md). Starter: `python exercises/ml/week-13/starter.py` from the repo root.

## 🤔 Reflection

1. Why does weight sharing beat “a separate weight per day” on images?
2. What does max-pool throw away, and when is that a feature?
3. Would you CNN a one-hot `plan_type`? Why not?

## 🔗 Next week

RNNs: the stencil starts to have **memory**. We walk the sequence left to right and carry a clipboard.
