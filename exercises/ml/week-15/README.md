# Exercise — Week 15 — Transformers: Everything Looks at Everything

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-15/starter.py
```

## ✍️ Exercises

**1. Remove positions.** Comment out `+ self.pos(pos)`. What happens to accuracy? That is the “order is invisible” lesson.

**2. Attention map.** After training, run one sentence through `self.enc.layers[0].self_attn` (or print `weights` from the toy block). Which characters attend to the `!`?

**3. Architecture memo.** Four sentences to your VP: CNN vs RNN vs Transformer vs GBT, with one CloudWave example each.
