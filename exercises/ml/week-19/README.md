# Exercise — Week 19 — RNNs: A Clipboard That Walks the Sequence

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-19/starter.py
```

## ✍️ Exercises

**1. Use the mean hidden state** instead of the last step (`out.mean(dim=1)`). Does a mid-sequence dip get more say?

**2. Reverse the weeks** (`torch.flip`). If accuracy dies, the model was using *order*, not just the total.

**3. One-sentence LSTM.** Explain a forget gate to a PM who has used a shopping cart.
