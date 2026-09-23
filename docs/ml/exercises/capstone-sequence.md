---
description: Deep-learning capstone exercises — build CNN, RNN, and transformer encoders over CloudWave event sequences, bake them off against the Week-13 GBT across seeds, and write an honest verdict.
---

# Exercises — Capstone — Does the Order of Events Beat the Row?

Weeks 18–20 drew the pictures. Here the three architectures meet a real question with a real bar: the Week-13 GBT on the aggregated row. Every network also gets that row, so the only thing it can add is *order*.

## What you are building

Three encoders over each customer's event sequence, a three-seed bake-off against the GBT and a static-only control, one experiment that tests whether order matters, and a verdict you'd sign.

## Predict before you run

1. A customer's sequence as of 2024-06-01 has a median of how many events?
2. Rank the five models by PR-AUC before you train any of them: GBT, static-only MLP, CNN, GRU, transformer.
3. If you shuffle each test sequence's event order, what happens to a model that genuinely uses order?
4. How many of the 80 top-scored customers will actually churn — and is a difference of one or two between models a result?

## Before you start

- CPU only. The full bake-off (5 models × 3 seeds) takes about a minute.
- Finish Weeks 13 and 18–20 first. `capstone_sequence/` gives you the sequences, the harness, and the GBT; the starter gives you the shared `Steps`, `Head`, and `StaticOnly` modules.
- Every encoder's `forward` is `(tokens, recency, mask, static) -> logits`. Sequences are **left-padded**: position `-1` is always the latest real event.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/capstone-sequence/starter.py
pytest tests/test_capstone_sequence.py
```

**1. The detector.** Write `ConvEncoder`: a `Conv1d` over `Steps`, max-pooled over **real** positions only, then `Head(D + n_static)`.

??? tip "Hint 1 — a nudge"
    Week 18: one stencil, many places, keep the loudest hit. But most rows are mostly padding — what would a plain max pool pick up from the padded positions?

??? tip "Hint 2 — the approach"
    `Conv1d` wants `(batch, channels, steps)`, so transpose `Steps`' output and back. Before the max, `masked_fill` padded positions with `-inf` so they can never win. Pass the pooled vector *and* `static` to the head.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    import torch
    import torch.nn as nn

    from capstone_sequence.data import MAX_LEN, PAD, VOCAB_SIZE, bakeoff_data
    from capstone_sequence.harness import bakeoff, fit_predict, metrics

    D = 16
    N_TRAIN, SEEDS, EPOCHS = 4000, (0, 1), 10  # raise to 8000, (0, 1, 2), 20 for the write-up


    class Steps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
            self.mix = nn.Linear(D + 1, D)

        def forward(self, tokens, recency):
            return torch.relu(self.mix(torch.cat([self.embed(tokens), recency.unsqueeze(-1)], dim=-1)))


    class Head(nn.Module):
        def __init__(self, n_in: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, *parts):
            return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


    class StaticOnly(nn.Module):
        def __init__(self, n_static: int) -> None:
            super().__init__()
            self.head = Head(n_static)

        def forward(self, tokens, recency, mask, static):
            return self.head(static)


    class ConvEncoder(nn.Module):
        def __init__(self, n_static: int) -> None:
            super().__init__()
            self.steps = Steps()
            self.conv = nn.Conv1d(D, D, kernel_size=3, padding=1)
            self.head = Head(D + n_static)

        def forward(self, tokens, recency, mask, static):
            h = torch.relu(self.conv(self.steps(tokens, recency).transpose(1, 2))).transpose(1, 2)
            pooled = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            return self.head(pooled, static)


    train, test = bakeoff_data(n_train=N_TRAIN)
    print(ConvEncoder(train.static.shape[1])(*[torch.from_numpy(a[:4]) for a in
                                              (train.tokens, train.recency, train.mask, train.static)]))
    ```

**2. The clipboard.** Write `GRUEncoder`: a `GRU` over `Steps`, read at the latest event.

??? tip "Hint 1 — a nudge"
    Week 19: the clipboard after the final step carries the whole walk. With left padding, where is the final *real* step for every row?

??? tip "Hint 2 — the approach"
    `nn.GRU(D, D, batch_first=True)` returns `(outputs, h_n)`. Because every row ends on a real event, `outputs[:, -1, :]` is the state after that customer's latest event — no packing needed. Concatenate with `static` in the head.

??? example "Hint 3 — most of the code"
    ```python
    class GRUEncoder(nn.Module):
        def __init__(self, n_static: int) -> None:
            super().__init__()
            self.steps = Steps()
            self.rnn = nn.GRU(D, D, batch_first=True)
            self.head = Head(D + n_static)

        def forward(self, tokens, recency, mask, static):
            out, _ = self.rnn(self.steps(tokens, recency))
            return self.head(out[:, -1, :], static)
    ```

**3. Everything looks at everything.** Write `AttentionEncoder`: learned positions, a one-layer `TransformerEncoder` that ignores padding, and a masked mean-pool.

??? tip "Hint 1 — a nudge"
    Week 20: without positions, attention sees a bag of events. And without a padding mask, every real event spends attention on blank slots. Which argument tells the encoder which slots are blank?

??? tip "Hint 2 — the approach"
    Add `nn.Embedding(MAX_LEN, D)` of `torch.arange(MAX_LEN)` to the steps. `TransformerEncoderLayer(D, nhead=2, dim_feedforward=32, batch_first=True)`, and pass `src_key_padding_mask=~mask` (True means *ignore*). Pool with the mask: sum real positions, divide by their count.

??? example "Hint 3 — most of the code"
    ```python
    class AttentionEncoder(nn.Module):
        def __init__(self, n_static: int) -> None:
            super().__init__()
            self.steps = Steps()
            self.pos = nn.Embedding(MAX_LEN, D)
            layer = nn.TransformerEncoderLayer(D, nhead=2, dim_feedforward=32, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.head = Head(D + n_static)

        def forward(self, tokens, recency, mask, static):
            x = self.steps(tokens, recency) + self.pos(torch.arange(tokens.shape[1]))
            h = self.encoder(x, src_key_padding_mask=~mask)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return self.head(pooled, static)
    ```

**4. The bake-off.** Run all five models over three seeds. Read the table with its standard deviations.

??? tip "Hint 1 — a nudge"
    One seed's AUC is an anecdote. Which gaps in your table are bigger than the seed-to-seed spread — and which model is the *control*?

??? tip "Hint 2 — the approach"
    `bakeoff(encoders, train, test, seeds=...)` returns mean and sd per model, plus PR-AUC lift over the dummy and mean hits in the top 80. Compare each encoder with **the static-only MLP**, not just the GBT: that isolates what the sequence added.

??? example "Hint 3 — most of the code"
    ```python
    encoders = {
        "mlp, static only": StaticOnly,
        "cnn + static": ConvEncoder,
        "gru + static": GRUEncoder,
        "transformer + static": AttentionEncoder,
    }
    table = bakeoff(encoders, train, test, seeds=SEEDS, epochs=EPOCHS)
    print(table.to_string())
    ```

**5. Does order matter?** Pick your best sequence model and score the test set twice: as-is, and with each customer's real events shuffled into a random order. Report both AUCs.

??? tip "Hint 1 — a nudge"
    If a model uses *order*, scrambling order should hurt it. If scores barely move, what was it actually using?

??? tip "Hint 2 — the approach"
    Keep padding where it is; permute only the real positions in each row, moving `tokens` and `recency` together. Train once with `fit_predict`, then call the trained model on both versions — or train twice with the same seed and score each version.

??? example "Hint 3 — most of the code"
    ```python
    from dataclasses import replace


    def shuffle_real_events(split, seed: int = 0):
        rng = np.random.default_rng(seed)
        tokens, recency = split.tokens.copy(), split.recency.copy()
        for i in range(len(tokens)):
            real = np.flatnonzero(split.mask[i])
            order = rng.permutation(real)
            tokens[i, real], recency[i, real] = tokens[i, order], recency[i, order]
        return replace(split, tokens=tokens, recency=recency)


    scrambled = shuffle_real_events(test)
    print("as-is    ", metrics(fit_predict(GRUEncoder, train, test, epochs=EPOCHS), test.y)["auc"])
    print("scrambled", metrics(fit_predict(GRUEncoder, train, scrambled, epochs=EPOCHS), test.y)["auc"])
    ```
    What the gap — or its absence — means is yours to write.

**6. The verdict.** Four sentences for Marcus, who read a blog post about transformers: what you compared, what won, what the sequence added, and what data would change your answer. Then judge your best network and the GBT through one brief from the [scenario bank](../capstone-scenarios.md).

??? tip "Hint 1 — a nudge"
    The honest verdict can be "the row wins." That's a result, not a failure — as long as the table and the seed spread back it up.

??? tip "Hint 2 — the approach"
    Lead with the control comparison: sequence model vs static-only MLP says what order added. Then the GBT comparison. For the brief, `select(test.frame, scores, brief)` and `judge(...)` work on any model's scores — the brief doesn't care which model made them.

??? example "Hint 3 — a skeleton"
    ```text
    Compared: GBT, a static-only MLP, and CNN / GRU / transformer encoders that also saw the row — <n> seeds, backtest.
    Won:      <model> at PR-AUC <x> ± <sd>; the gap to <runner-up> is <bigger / smaller> than the seed spread.
    Sequence: adding event order <helped / did nothing / hurt> by <Δ> against the static-only control, and scrambling order <…>.
    Changes my mind: <longer histories? event types that mean what they say? a different question?>.
    ```

## Success criteria

- Three encoders that train without NaNs (every row, including customers with no events).
- A three-seed table with standard deviations and the static-only control.
- A scrambled-order result for your best sequence model.
- A four-sentence verdict that cites the control, the spread, and one brief.

## After you run

`pytest tests/test_capstone_sequence.py` checks the sequence invariants (no future events, left padding, a `NONE` token for silent customers) and runs the reference encoders. Compare your table with the reference's — then look at which numbers moved between seeds more than between models.

## Lesson link

[Capstone — Does the Order of Events Beat the Row?](../capstone-sequence.md)
