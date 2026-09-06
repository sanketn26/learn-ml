---
description: Pool an RNN's hidden states, reverse the input sequence to test whether the model relies on order, and explain an LSTM forget gate in plain terms.
---

# Exercises — Week 19 — RNNs: A Clipboard That Walks the Sequence

## What you are building

A mean-pooled hidden state, a reversed-week run, and a one-sentence forget-gate explanation.

## Predict before you run

1. Does `out.mean(dim=1)` give a mid-sequence dip more say than the last step?
2. If `torch.flip` kills accuracy, was the model using order or the total?
3. What does a forget gate throw away, in shopping-cart language?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-19/starter.py
```

**1. Use the mean hidden state** instead of the last step (`out.mean(dim=1)`). Does a mid-sequence dip get more say?

**2. Reverse the weeks** (`torch.flip`). If accuracy dies, the model was using *order*, not just the total.

**3. One-sentence LSTM.** Explain a forget gate to a PM who has used a shopping cart.

## Success criteria

- Mean vs last-step comparison.
- Flipped-sequence result interpreted.
- One PM sentence on the forget gate.

## Debugging clues

- Last hidden state ignores early weeks unless the clipboard carried them.
- If flip does nothing, you were summing.
- CPU is enough; this is not a language model.

## After you run

An RNN is a clipboard that walks. Transformers (next week) jump instead of walking.

## Lesson link

[Week 19 — RNNs: A Clipboard That Walks the Sequence](../week-19.md)
