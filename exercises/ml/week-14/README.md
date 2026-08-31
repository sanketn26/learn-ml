# Exercise — Week 14 — Neural Nets, Without the Mystique

## What you are building

A linear MLP vs logistic regression, an oversized net, a five-line VP memo, and a training loop with `zero_grad` commented out.

## Predict before you run

1. With `activation="identity"`, will MLP AUC rhyme with logistic regression?
2. `(128, 128, 128)` on this table: train AUC vs test?
3. If you skip `opt.zero_grad()`, does loss explode, freeze, or look fine?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-14/starter.py
```

**1. Linear MLP.** Set `activation="identity"` (no ReLU). Compare AUC to logistic regression. They should rhyme.

**2. Too much net.** `hidden_layer_sizes=(128, 128, 128)` on this data. What happens to train vs test?

**3. Decision memo.** Write five lines to your VP: why CloudWave's churn model will stay a GBT this quarter.

**4. Break the loop.** Comment out `opt.zero_grad()` and rerun 5 epochs. What happens to the loss? (It should explode or thrash.)

## Success criteria

- Identity MLP vs logreg AUCs.
- Deep-net train/test gap.
- Five-line memo.
- Loss behavior without `zero_grad`.

## Debugging clues

- sklearn MLP vs a hand-rolled torch loop are both fair for (1)/(2) vs (4).
- CPU is enough. Do not reach for CUDA.
- Accumulated grads without zeroing are a running sum, not “momentum.”

## After you run

A net is mixers + switches + a four-line training step. On this SaaS table the GBT still ships.

## Lesson link

[Week 14 — Neural Nets, Without the Mystique](../../../docs/ml/week-14.md)
