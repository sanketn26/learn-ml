# Exercises — Week 20 — Transformers: Everything Looks at Everything

Do these after reading [Week 20 — Transformers: Everything Looks at Everything](../week-20.md).

**1. Remove positions.** Comment out `+ self.pos(pos)`. What happens to accuracy? That is the “order is invisible” lesson.

**2. Attention map.** Use the 3-token toy (`login` / `failed` / `again`) from the lesson — print `weights`. Which token does `"failed"` look at? Do not poke `self.enc.layers[0].self_attn` on raw character ids; embed first or stay with the toy.

**3. Architecture memo.** Four sentences to your VP: CNN vs RNN vs Transformer vs GBT, with one CloudWave example each.
