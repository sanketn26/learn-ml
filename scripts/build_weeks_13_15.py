#!/usr/bin/env python3
"""Weeks 13–15 — CNNs, RNNs, Transformers in PyTorch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbformat_util import BOOT, LAPTOP_BOX, code_cell, md_cell, write_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"

TORCH = '''
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Install with:  pip install torch") from exc

torch.manual_seed(0)
DEVICE = torch.device("cpu")
print("torch", torch.__version__, "device", DEVICE)
'''


def week13():
    cells = [
        md_cell(
            """# Week 13 — CNNs: Sliding Detectors

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

<div class="think-box">
<strong>Think of it like… Ctrl+F with a fuzzy stencil, or a antivirus signature scan.</strong>
<p>You do not write a separate rule for “spike on Monday” and “spike on Thursday.” You write <em>one</em> “usage spike” detector and slide it along the week. The detector’s numbers are learned. That reuse is why CNNs have so few weights compared to a giant dense net on every pixel.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(BOOT + "\n" + TORCH),
        md_cell(
            """## Visual: a 1-D filter sliding

```
usage by day:   [2, 2, 8, 9, 2, 2, 3]
detector:          [−1, 2, −1]     “a spike relative to neighbors”
slide:          pos0: 2·-1 + 2·2 + 8·-1 = -6
                pos1: 2·-1 + 8·2 + 9·-1 =  5   ← hit
                pos2: 8·-1 + 9·2 + 2·-1 =  8   ← hit
```

A **2-D CNN** is the same idea on a grid (a screenshot, a heatmap, an MRI). Same stencil, two axes.

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p><code>nn.Conv1d</code> is a 1-D convolution over time. <code>nn.Conv2d</code> is over height×width. <code>in_channels</code> is “how many parallel traces” (RGB = 3, or many features per day). <code>out_channels</code> is “how many different detectors we learn.” Pooling is <code>max</code> in a window — keep the strongest hit, throw away the exact timestamp.</p>
</div>
"""
        ),
        code_cell(
            """# Hand-built spike detector — no learning yet
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
"""
        ),
        md_cell(
            """## CloudWave: usage as a short time series

Each user becomes a length-T vector of weekly usage. The CNN’s job: “does this *shape* look like someone about to churn?” — not “what is their total.”
"""
        ),
        code_cell(
            """from course_style import load_weekly_usage_grid

X, y = load_weekly_usage_grid(DATA)  # ~3k users × 12 weeks, CPU-friendly
print(f"users={len(X):,}  timesteps={X.shape[1]}  churn={y.mean():.3f}")

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
tr, te = idx[:cut], idx[cut:]
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
"""
        ),
        md_cell(
            """## A tiny Conv1d in PyTorch

`input` shape for Conv1d is `(batch, channels, time)`. We have one channel (usage).
"""
        ),
        code_cell(
            """class UsageCNN(nn.Module):
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
"""
        ),
        md_cell(
            """## 2-D picture (so “CNN” in papers makes sense)

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

<div class="watch-box">
<strong>Watch out</strong>
<p>A CNN on 12 weekly totals is a teaching toy. Real CNNs earn their keep on <em>grids</em> (product screenshots, document scans) or long dense signals. For CloudWave’s 7-column table, last week’s GBT is still the right ship.</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Ship a CNN when nearby positions <em>mean the same thing</em> (pixels, audio samples, equally spaced sensors). Do not ship one because a blog said “deep learning.”</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Kernel size.** Change `kernel_size` to 5. Does test accuracy move? What did you make the detector look at?

**2. Dense baseline.** Flatten the 12 weeks into a `nn.Linear(12, 1)` and compare. If they tie, the *shape* was not the signal — the *total* was.

**3. Draw it.** Sketch one user as 12 boxes and a 3-wide stencil in three positions. Circle the position you think fires on a late-week drop.

## 🤔 Reflection

1. Why does weight sharing beat “a separate weight per day” on images?
2. What does max-pool throw away, and when is that a feature?
3. Would you CNN a one-hot `plan_type`? Why not?

## 🔗 Next week

RNNs: the stencil starts to have **memory**. We walk the sequence left to right and carry a clipboard.
"""
        ),
    ]
    write_notebook(OUT / "week-13-saas.ipynb", cells, "Week 13 — CNNs")


def week14():
    cells = [
        md_cell(
            """# Week 14 — RNNs: A Clipboard That Walks the Sequence

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

<div class="think-box">
<strong>Think of it like… a running total, or a state machine.</strong>
<p>You walk a user’s weeks from oldest to newest. After each week you update a small vector — the <strong>hidden state</strong> — the way a cashier updates the subtotal. At the end, that vector is “everything I still remember about this customer.” A linear layer turns it into a churn score.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(BOOT + "\n" + TORCH),
        md_cell(
            """## Unrolling the loop

```
h0 = zeros
h1 = f(h0, week1)
h2 = f(h1, week2)
...
hT = f(hT-1, weekT)  →  score

f is THE SAME function every step. That is the whole trick.
```

<div class="math-box">
<strong>Math, translated — vanishing memory</strong>
<p>Training walks backward through those T steps. If each step multiplies the “how much should I remember?” signal by a number &lt; 1, after 30 steps the early weeks have been multiplied into dust. The model becomes “whatever happened last.” <strong>LSTMs / GRUs</strong> add gates: learned locks that can say “keep this bit unchanged.” They do not magically remember 10,000 steps. They just forget slower.</p>
</div>
"""
        ),
        code_cell(
            """# Memory fade cartoon: multiply a signal by 0.7, 30 times
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
"""
        ),
        md_cell(
            """## Same CloudWave sequences as Week 13, new `forward`
"""
        ),
        code_cell(
            """from course_style import load_weekly_usage_grid

X, y = load_weekly_usage_grid(DATA, random_state=1)
rng = np.random.default_rng(1)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
Xtr, Xte = X[idx[:cut]], X[idx[cut:]]
ytr, yte = y[idx[:cut]], y[idx[cut:]]
print(Xtr.shape, "churn", float(ytr.mean()))
"""
        ),
        md_cell(
            """## Vanilla RNN, then a GRU

`nn.RNN` / `nn.GRU` want `(batch, time, features)` when `batch_first=True`.
"""
        ),
        code_cell(
            """class SequenceNet(nn.Module):
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
"""
        ),
        md_cell(
            """## LSTM / GRU gates, in English

| Gate | Plain English |
|---|---|
| Forget / reset | “Throw this bit of the clipboard away” |
| Input / update | “Write the new week in” |
| Output | “What part of the clipboard is the answer *this* step” |

You do not tune gates by hand. The training loop learns when to lock.

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>An RNN is a <code>for</code> loop you can backprop through. That loop is <strong>sequential</strong> — week 7 cannot start until week 6 is done — so GPUs hate long RNNs. Transformers (next week) look at every week at once. That is why industry moved.</p>
</div>

<div class="watch-box">
<strong>Watch out</strong>
<p>Teacher forcing, packed sequences, bidirectional RNNs — you will see the words. They are implementation details around the same clipboard. Do not start there. Start with “last hidden state → linear.”</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>A small GRU is still fine for <em>short</em> sensor traces and on-device models. For language, search, and anything longer than a few dozen steps, ship a Transformer (or call an API that already is one).</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Use the mean hidden state** instead of the last step (`out.mean(dim=1)`). Does a mid-sequence dip get more say?

**2. Reverse the weeks** (`torch.flip`). If accuracy dies, the model was using *order*, not just the total.

**3. One-sentence LSTM.** Explain a forget gate to a PM who has used a shopping cart.

## 🤔 Reflection

1. How is an RNN like a fold/reduce? How is it not (the body is learned)?
2. Why did vanishing gradients hurt signup-week features more than last-week features?
3. Why are RNNs slow on a GPU compared to a CNN or a Transformer?

## 🔗 Next week

Transformers: throw away the clipboard. Every week (or word) looks at every other one, in parallel.
"""
        ),
    ]
    write_notebook(OUT / "week-14-saas.ipynb", cells, "Week 14 — RNNs")


def week15():
    cells = [
        md_cell(
            """# Week 15 — Transformers: Everything Looks at Everything

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a search index, a join, or `dict.get`. This is the architecture behind GPT, BERT, Copilot, and most of LangChain.

---

## 🎯 What you will be able to do

- Explain **attention** as a soft dictionary lookup: query → keys → weighted values
- See why we add **position** (the model has no loop, so it cannot “know” order otherwise)
- Build a tiny self-attention block in PyTorch and watch weights light up
- Classify CloudWave **feedback text** with a small Transformer encoder
- Know encoder vs decoder vs “the API you will actually call”

<div class="think-box">
<strong>Think of it like… a database lookup where every row is a candidate, and the score is “how related are you to my question?”</strong>
<p><strong>Query (Q)</strong> = what this token is looking for.<br>
<strong>Keys (K)</strong> = what every token advertises it contains.<br>
<strong>Values (V)</strong> = the payload you actually mix in if the key matched.</p>
<p>Attention weights are a softmax over “how well does my query match each key.” Then you take the weighted sum of values. No clipboard. No left-to-right bottleneck. Every token does this <em>in parallel</em>.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(BOOT + "\n" + TORCH),
        md_cell(
            """## The picture

```
tokens:     [  "login" , "failed" , "again" ]
               Q K V      Q K V      Q K V
                 \\         |         /
                  \\        |        /
                   softmax(Q · Kᵀ)   ← who should I read?
                         │
                    mix of V's
```

<div class="math-box">
<strong>Math, translated</strong>
<p><code>weights = softmax(Q @ K.T / sqrt(d))</code> → a row of positive numbers that sum to 1, one row per token. Divide by <code>sqrt(d)</code> so the dot products do not explode when the vectors are long. Then <code>output = weights @ V</code>. That is attention. Multi-head = several of these lookups in parallel, then concatenated — several reviewers reading for different things.</p>
</div>
"""
        ),
        code_cell(
            """# Tiny self-attention you can print
torch.manual_seed(0)
tokens = ["login", "failed", "again"]
d = 4
X = torch.randn(len(tokens), d)          # pretend embeddings
Wq = torch.randn(d, d); Wk = torch.randn(d, d); Wv = torch.randn(d, d)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / d ** 0.5
weights = torch.softmax(scores, dim=-1)
out = weights @ V

fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(weights.detach().numpy(), cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(3), tokens); ax.set_yticks(range(3), tokens)
ax.set_xlabel("looking at"); ax.set_ylabel("token")
ax.set_title("Attention weights (rows sum to 1)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
print("Each row is a probability distribution over who to read.")
"""
        ),
        md_cell(
            """## Why position encodings exist

A Transformer is a bag of lookups. `"the movie was not good"` and `"the movie was good not"` look the same unless you **stamp** each token with where it sat.

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Position = an extra feature, like adding <code>index</code> to a log line before you embed it. Modern models use rotary / learned positions. You do not need the sine formula. You need: <em>without a position stamp, order is invisible.</em></p>
</div>
"""
        ),
        md_cell(
            """## CloudWave: classify feedback text

We will bag-of-words the comments into a short token id sequence and run a toy encoder. This is **not** BERT. It is the moving parts, small enough to train on a laptop in a minute.
"""
        ),
        code_cell(
            """feedback = pd.read_json(DATA / "feedback.json", lines=True)
# Binary: praise vs everything else (or bug vs not)
feedback["y"] = (feedback["category"].str.lower() == "praise").astype(int)
print(feedback["category"].value_counts().head())
print("praise rate", feedback["y"].mean().round(3))

# Character-level tokens — ugly, honest, no extra downloads
def encode(text: str, n=32):
    ids = [min(ord(c), 126) for c in str(text).lower()[:n]]
    ids += [0] * (n - len(ids))
    return ids

# 4k comments is enough to see the loop move — the rest is the same idea
feedback = feedback.sample(n=min(4000, len(feedback)), random_state=0)
ids = np.array([encode(t) for t in feedback["feedback_text"]], dtype=np.int64)
y = feedback["y"].to_numpy(dtype=np.int64)
rng = np.random.default_rng(0)
idx = rng.permutation(len(ids))
cut = int(0.8 * len(ids))
Xtr, Xte = ids[idx[:cut]], ids[idx[cut:]]
ytr, yte = y[idx[:cut]], y[idx[cut:]]
print("seq shape", Xtr.shape, "vocab 0–126")
"""
        ),
        code_cell(
            """class TinyTransformer(nn.Module):
    def __init__(self, vocab=127, d=24, nhead=4, ntok=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ntok, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                           dim_feedforward=48, batch_first=True,
                                           dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d, 1)

    def forward(self, token_ids):
        b, t = token_ids.shape
        pos = torch.arange(t).unsqueeze(0).expand(b, t)
        x = self.emb(token_ids) + self.pos(pos)
        h = self.enc(x)                       # (B, T, d)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


model = TinyTransformer()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
print("params", sum(p.numel() for p in model.parameters()))

def batch(X, y, bs=256):
    for i in range(0, len(X), bs):
        yield torch.tensor(X[i:i+bs]), torch.tensor(y[i:i+bs], dtype=torch.float32)

hist = []
for epoch in range(4):
    model.train()
    tr_loss = 0.0
    n = 0
    for xb, yb in batch(Xtr, ytr):
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += float(loss) * len(xb); n += len(xb)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte))
        te_loss = float(F.binary_cross_entropy_with_logits(
            logits, torch.tensor(yte, dtype=torch.float32)))
        acc = float(((logits.sigmoid() > 0.5).numpy() == yte).mean())
    hist.append((tr_loss / n, te_loss, acc))
    print(f"epoch {epoch+1}  train {hist[-1][0]:.3f}  test {te_loss:.3f}  acc {acc:.3f}")

hist = np.array(hist)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(hist[:, 0], label="train loss")
ax.plot(hist[:, 1], label="test loss")
ax.set_title("Toy Transformer on feedback text")
ax.legend(); plt.tight_layout(); plt.show()
print("majority acc", 1 - yte.mean())
"""
        ),
        md_cell(
            """## Encoder vs decoder vs the API

| Shape | What it does | You have used it as |
|---|---|---|
| **Encoder** (this week) | Read a whole sequence, emit a representation | BERT, embedding models, Week 4 RAG later |
| **Decoder** | Generate the next token, one at a time, looking left | GPT, chat models |
| **Encoder–decoder** | Read a source, write a target | translation, summarization |

The four-line training step is unchanged. The `forward` is “stack of attention + feed-forward + residual skip,” repeated.

<div class="cue-box">
<strong>Visual cue — residual skip</strong>
<p><code>x = x + attention(x)</code>. Same idea as a git commit on top of the previous tree: keep the old signal, add a delta. That is why 96-layer models can still train.</p>
</div>

<div class="watch-box">
<strong>Watch out</strong>
<p>This notebook is a teaching Transformer, not a product. Do not scrape a tiny encoder and call it “we built GPT.” Production language models are pretrained on a planet of text. Your job is usually: pick a model, prompt it, fine-tune lightly, or embed + retrieve (the LangChain course).</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<ul>
<li><strong>Tabular churn</strong> → GBT (Week 10–12).</li>
<li><strong>Screenshots / dense grids</strong> → CNN (Week 13).</li>
<li><strong>Short sensor traces on-device</strong> → GRU maybe (Week 14).</li>
<li><strong>Language, code, mixed documents</strong> → Transformer, usually via an API or a small open model — not from-scratch on 10k comments.</li>
</ul>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Remove positions.** Comment out `+ self.pos(pos)`. What happens to accuracy? That is the “order is invisible” lesson.

**2. Attention map.** After training, run one sentence through `self.enc.layers[0].self_attn` (or print `weights` from the toy block). Which characters attend to the `!`?

**3. Architecture memo.** Four sentences to your VP: CNN vs RNN vs Transformer vs GBT, with one CloudWave example each.

## 🤔 Reflection

1. Attention is a join. What are the two tables?
2. Why can a Transformer use a GPU better than an RNN?
3. After this week, what is left that is *not* “just attention”? (tokenization, alignment, eval, product)

## 🎓 You now have the three pillars

| Pillar | Where |
|---|---|
| **Strong Python + NumPy + Pandas + PyTorch** | Weeks 0–2, 11 |
| **ML fundamentals** (regression, classification, overfit, bias, variance) | Weeks 6–10 |
| **Deep learning** (nets, CNN, RNN, Transformer, the training loop) | Weeks 11, 13–15 |

The LangChain course is what you do when the Transformer *already exists* and you need to wire it into a product. You are ready for it.
"""
        ),
    ]
    write_notebook(OUT / "week-15-saas.ipynb", cells, "Week 15 — Transformers")


if __name__ == "__main__":
    week13()
    week14()
    week15()
