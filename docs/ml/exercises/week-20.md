# Exercises — Week 20 — Transformers: Everything Looks at Everything

## What you are building

A no-position run, a 3-token attention map, and a four-sentence architecture memo for a VP.

## Predict before you run

1. If you comment out `+ self.pos(pos)`, does accuracy fall because order became invisible?
2. In the toy `login` / `failed` / `again`, which token does `"failed"` look at?
3. For CloudWave churn on a 7-column table, who wins: CNN, RNN, Transformer, or GBT?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-20/starter.py
```

**1. Remove positions.** Comment out `+ self.pos(pos)`. What happens to accuracy? That is the “order is invisible” lesson.

**2. Attention map.** Use the 3-token toy (`login` / `failed` / `again`) from the lesson — print `weights`. Which token does `"failed"` look at? Do not poke `self.enc.layers[0].self_attn` on raw character ids; embed first or stay with the toy.

**3. Architecture memo.** Four sentences to your VP: CNN vs RNN vs Transformer vs GBT, with one CloudWave example each.

## Success criteria

- Accuracy with/without positions.
- Printed weights on the 3-token toy.
- Four-sentence memo.

## Debugging clues

- Attention on raw char ids is nonsense — embed or use the toy.
- This week does not train GPT from scratch.
- A GBT still wins on the Customer 360 table.

## After you run

Attention is a soft join. You can read a block diagram. You should not expect to implement FlashAttention or debug CUDA kernels.

## Lesson link

[Week 20 — Transformers: Everything Looks at Everything](../week-20.md)
