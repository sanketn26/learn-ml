---
description: Learn Transformer self-attention as a soft dictionary lookup with queries, keys, and values, the architecture behind GPT and BERT.
---

# Week 20 — Transformers: Everything Looks at Everything

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a search index, a join, or `dict.get`. This is the architecture behind GPT, BERT, Copilot, and most of LangChain.

---

## 🎯 What you will be able to do

- Explain **attention** as a soft dictionary lookup: query → keys → weighted values
- See why we add **position** (the model has no loop, so it cannot “know” order otherwise)
- Build a tiny self-attention block in PyTorch and watch weights light up
- Classify CloudWave **feedback text** with a small Transformer encoder
- Know encoder vs decoder vs “the API you will actually call”

!!! think "Think of it like… a database lookup where every row is a candidate, and the score is “how related are you to my question?”"

    **Query (Q)** = what this token is looking for.

    **Keys (K)** = what every token advertises it contains.

    **Values (V)** = the payload you actually mix in if the key matched.

    Attention weights are a softmax over *scaled* query–key match: divide the dots by √dₖ so long vectors do not explode, then mix values. No clipboard. No left-to-right bottleneck. Every token does this *in parallel*.

    Canonical: **Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V** (Vaswani et al., *Attention Is All You Need*).

## If you already write software

A Transformer has **no loop**. Every token looks at every other token, in parallel, and decides who matters. That “who matters” is **attention**.

Think of it as a **soft join**. **Mental model:** a query probes keys and mixes in values. **Simplification:** SQL has no scale factor. The similarity that actually ships is **scaled** dot-product attention.

```
SQL                             Attention
──────────────────────────      ─────────────────────────────
probe row (query)               Q  — what am I looking for?
table keys                      K  — what does each token advertise?
table values                    V  — what does each token actually carry?
JOIN ON similarity              softmax(QKᵀ / √dₖ)  — a distribution over partners
SELECT values                   weighted sum of V
```

Pipeline (same thing, left to right):

```
QKᵀ  →  divide by √dₖ  →  softmax  →  weights  →  weighted V
```

Canonical equation: **Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V** (Vaswani et al., *Attention Is All You Need*). Masking, multi-head, and rotary positions wrap that same scaled-dot core; they do not drop `/ √dₖ`.

Because there is no left-to-right clipboard, the model does not know order unless you **add positions** (positional encodings). That is the whole trick: attention for content, positions for order.

### What you are not building

This week’s tiny encoder on feedback text is a teaching Transformer. It is not GPT. Production language models are pretrained on a planet of text. Your job in a SaaS company is usually:

1. pick a model
2. prompt it
3. maybe fine-tune lightly
4. or embed + retrieve (the LangChain course)

Do not scrape a 4-layer encoder and tell the board you built a foundation model.

### Picture “everything looks at everything”

```
"billing"   looks at  "broken", "invoice", "twice"   → support-billing
"love"      looks at  "dashboard", "fast"            → praise
```

The same word “charge” means different things next to “battery” vs “credit card.” Attention is how the model does that disambiguation without a hand-written parser.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all ~49k rows.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

## The picture

```
tokens:     [  "login" , "failed" , "again" ]
               Q K V      Q K V      Q K V
                 \         |         /
                  \        |        /
         QKᵀ → ÷ √dₖ → softmax → weights → mix of V
                         ↑
                  who should I read?
```

!!! math "Math, translated"

    Canonical (Vaswani et al., *Attention Is All You Need*):

    **Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V**

    Pipeline: `QKᵀ` → divide by `√dₖ` → softmax → weights → weighted `V`.

    In code: `weights = softmax(Q @ K.T / sqrt(d_k))` → a row of positive numbers that sum to 1, one row per token. Divide by `sqrt(d_k)` so the dot products do not explode when the vectors are long. Then `output = weights @ V`. That is attention. Multi-head = several of these lookups in parallel, then concatenated — several reviewers reading for different things.

`Q`, `K`, `V` are not three different pieces of the token — they are the **same embedding `X`**, run through three learned weight matrices: `Q = XW_Q`, `K = XW_K`, `V = XW_V`. Training is what decides what each matrix pays attention to.

```
representation X

        ┌── W_Q → Q = XW_Q  (what am I searching for?)
X ──────┼── W_K → K = XW_K  (what do I match against?)
        └── W_V → V = XW_V  (what information do I contribute?)
```

## Before you run this

Predict:

1. Will each row of `weights` sum to 1?
2. Will `"failed"` look mostly at itself, or at `"login"` / `"again"`?
3. Why divide the scores by `sqrt(d)` before softmax — what would go wrong without it? Check the unscaled panel against the scaled one.

## Run it

Compare both heatmaps with your prediction.

## Explain the difference

If your prediction was wrong, what assumption was wrong?

```python
# Tiny self-attention you can print
torch.manual_seed(0)
tokens = ["login", "failed", "again"]
d = 4
X = torch.randn(len(tokens), d)          # pretend embeddings
Wq = torch.randn(d, d); Wk = torch.randn(d, d); Wv = torch.randn(d, d)  # W_Q, W_K, W_V
Q, K, V = X @ Wq, X @ Wk, X @ Wv
d_k = d
scores_raw = Q @ K.T
scores = scores_raw / d_k ** 0.5  # scale: QKᵀ / √dₖ
weights_raw = torch.softmax(scores_raw, dim=-1)
weights = torch.softmax(scores, dim=-1)
out = weights @ V

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4))
for ax, W, title in (
    (axes[0], weights_raw, "Unscaled softmax(QKᵀ)"),
    (axes[1], weights, "Scaled softmax(QKᵀ / √dₖ)"),
):
    im = ax.imshow(W.detach().numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3), tokens)
    ax.set_yticks(range(3), tokens)
    ax.set_xlabel("looking at")
    ax.set_ylabel("token")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
print("Each row is a probability distribution over who to read.")
print("Without /√dₖ the unscaled panel saturates; scaled stays usable.")
```

## Why position encodings exist

A Transformer is a bag of lookups. `"the movie was not good"` and `"the movie was good not"` look the same unless you **stamp** each token with where it sat.

!!! engineer "Engineer mental model"

    Position = an extra feature, like adding `index` to a log line before you embed it. Modern models use rotary / learned positions. You do not need the sine formula. You need: *without a position stamp, order is invisible.*

## CloudWave: classify feedback text

We will encode the comments as **character ids** (not bag-of-words) and run a toy encoder. This is **not** BERT. It is the moving parts, small enough to train on a laptop in a minute.

```python
feedback = pd.read_json(DATA / "feedback.json", lines=True)
# Binary: praise vs everything else. category now matches the text
# (praise rows actually say something nice).
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
```

```python
class TinyTransformer(nn.Module):
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
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        opt.step()
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
```

## Encoder vs decoder vs the API

| Shape | What it does | You have used it as |
|---|---|---|
| **Encoder** (this week) | Read a whole sequence, emit a representation | BERT, embedding models, [LangChain week 4](../langchain/week-04.md) RAG |
| **Decoder** | Generate the next token, one at a time, looking left | GPT, chat models |
| **Encoder–decoder** | Read a source, write a target | translation, summarization |

The four-line training step is unchanged. The `forward` is “stack of attention + feed-forward + residual skip,” repeated.

!!! tip "Visual cue — residual skip"

    `x = x + attention(x)`. Same idea as a git commit on top of the previous tree: keep the old signal, add a delta. That is why 96-layer models can still train.


!!! warning "Watch out"

    This lesson is a teaching Transformer, not a product. Do not scrape a tiny encoder and call it “we built GPT.” Production language models are pretrained on a planet of text. Your job is usually: pick a model, prompt it, fine-tune lightly, or embed + retrieve ([LangChain week 4](../langchain/week-04.md)).


!!! success "Ship / don’t ship"

    - **Tabular churn** → GBT (Week 13–12).

    - **Screenshots / dense grids** → CNN (Week 18).

    - **Short sensor traces on-device** → GRU maybe (Week 19).

    - **Language, code, mixed documents** → Transformer, usually via an API or a small open model — not from-scratch on 10k comments.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-20.md). Starter: `python exercises/ml/week-20/starter.py` from the repo root.

## 🤔 Reflection

1. Attention is a join. What are the two tables?
2. Why can a Transformer use a GPU better than an RNN?
3. After this week, what is left that is *not* “just attention”? (tokenization, alignment, eval, product)

## 🎓 You now have the three pillars

| Pillar | Where |
|---|---|
| **Strong Python + NumPy + Pandas + PyTorch** | Weeks 0–2, 14 |
| **ML fundamentals** (regression, classification, overfit, bias, variance) | Weeks 6–10 |
| **Deep learning** (nets, CNN, RNN, Transformer, the training loop) | Weeks 14, 18–20 |

## Before you leave

Try one item from [self-checks — weeks 18–20](self-checks.md#weeks-18-20-dl-pictures-optional). You should be able to read a block diagram — not implement FlashAttention.

The LangChain course is what you do when the Transformer *already exists* and you need to wire it into a product. You are ready for it.
