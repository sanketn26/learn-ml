---
description: Ablate positional encoding in a transformer and measure the change, inspect a self-attention map, and compare CNN, RNN, transformer, and GBT tradeoffs.
---

# Exercises — Week 20 — Transformers: Everything Looks at Everything

## What you are building

A no-position run, a 3-token attention map, and a four-sentence architecture memo for a VP.

## Predict before you run

1. If you comment out `+ self.pos(pos)`, does accuracy fall because order became invisible?
2. In the toy `login` / `failed` / `again`, which token does `"failed"` look at?
3. For CloudWave churn on a 7-column table, who wins: CNN, RNN, Transformer, or GBT?

## Before you start

- This week does not train GPT from scratch. CPU and a 4k-comment sample are enough.
- Attention on raw character ids is nonsense — embed first, or stay with the toy.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-20/starter.py
```

**1. Remove positions.** Comment out `+ self.pos(pos)`. What happens to accuracy?

??? tip "Hint 1 — a nudge"
    Without position embeddings, self-attention sees a *bag* of characters. Would "not great" and "great, not" look different to it? Would it matter for spotting praise?

??? tip "Hint 2 — the approach"
    Add a `use_pos` flag to the lesson's `TinyTransformer`, train both versions with the same seed and split, and compare test accuracy against the majority baseline. A small change is a finding too — ask what the task needs.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from lib.course_data import find_data_dir

    fb = pd.read_json(find_data_dir() / "feedback.json", lines=True).sample(4000, random_state=0)
    y = (fb["category"].str.lower() == "praise").to_numpy(dtype=np.int64)


    def encode(text: str, n: int = 32) -> list[int]:
        ids = [min(ord(c), 126) for c in str(text).lower()[:n]]
        return ids + [0] * (n - len(ids))


    ids = np.array([encode(t) for t in fb["feedback_text"]], dtype=np.int64)
    idx = np.random.default_rng(0).permutation(len(ids))
    cut = int(0.8 * len(ids))
    Xtr, Xte, ytr, yte = ids[idx[:cut]], ids[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


    class TinyTransformer(nn.Module):
        def __init__(self, use_pos: bool = True, vocab: int = 127, d: int = 24, ntok: int = 32):
            super().__init__()
            self.use_pos = use_pos
            self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(ntok, d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=48, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d, 1)

        def forward(self, token_ids):
            x = self.emb(token_ids)
            if self.use_pos:
                x = x + self.pos(torch.arange(token_ids.shape[1]).expand_as(token_ids))
            return self.head(self.enc(x).mean(dim=1)).squeeze(-1)


    def fit_eval(use_pos: bool, epochs: int = 4) -> float:
        torch.manual_seed(0)
        model = TinyTransformer(use_pos)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(epochs):
            model.train()
            for i in range(0, len(Xtr), 256):
                opt.zero_grad()
                logits = model(torch.tensor(Xtr[i:i + 256]))
                F.binary_cross_entropy_with_logits(logits, torch.tensor(ytr[i:i + 256], dtype=torch.float32)).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return float(((model(torch.tensor(Xte)).sigmoid() > 0.5).numpy() == yte).mean())


    print(f"with positions={fit_eval(True):.3f}  without={fit_eval(False):.3f}  majority={1 - yte.mean():.3f}")
    ```

**2. Attention map.** Use the 3-token toy (`login` / `failed` / `again`) from the lesson — print `weights`. Which token does `"failed"` look at? Do not poke `self.enc.layers[0].self_attn` on raw character ids; embed first or stay with the toy.

??? tip "Hint 1 — a nudge"
    `weights` is a 3×3 table. Each *row* is one token deciding how much to read from every token, and each row sums to 1.

??? tip "Hint 2 — the approach"
    Rebuild the lesson's toy (same seed), then print the `"failed"` row with its column labels. The weights come from random matrices — say what the row shows, not what "failed" *should* look at.

??? example "Hint 3 — most of the code"
    ```python
    torch.manual_seed(0)
    tokens = ["login", "failed", "again"]
    d = 4
    E = torch.randn(len(tokens), d)
    Wq, Wk, Wv = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d)
    weights = torch.softmax((E @ Wq) @ (E @ Wk).T / d ** 0.5, dim=-1)
    print(pd.DataFrame(weights.numpy(), index=tokens, columns=tokens).round(2))
    ```

**3. Architecture memo.** Four sentences to your VP: CNN vs RNN vs Transformer vs GBT, with one CloudWave example each.

??? tip "Hint 1 — a nudge"
    Each architecture assumes something about the shape of its input. What shape is each CloudWave dataset — a grid over time, an ordered stream, free text, a row of columns?

??? tip "Hint 2 — the approach"
    One sentence per model: what it's good at, then the CloudWave data that fits it. For the Customer 360 table, your Week 13–14 bake-off numbers are the evidence — cite them.

??? example "Hint 3 — a skeleton"
    ```text
    CNN:         <what it detects> — e.g. <CloudWave data>.
    RNN:         <what it carries> — e.g. <…>.
    Transformer: <what it relates> — e.g. <…>.
    GBT:         <what it's for> — e.g. <…>, and our numbers say <…>.
    ```

## Success criteria

- Accuracy with/without positions.
- Printed weights on the 3-token toy.
- Four-sentence memo.

## After you run

Attention is a soft join. You can read a block diagram. You should not expect to implement FlashAttention or debug CUDA kernels.

## Lesson link

[Week 20 — Transformers: Everything Looks at Everything](../week-20.md)
