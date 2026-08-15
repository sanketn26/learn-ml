# Exercises — Week 11 — Neural Nets, Without the Mystique

Do these after reading [Week 11 — Neural Nets, Without the Mystique](../week-11.md).

**1. Linear MLP.** Set `activation="identity"` (no ReLU). Compare AUC to logistic regression. They should rhyme.

**2. Too much net.** `hidden_layer_sizes=(128, 128, 128)` on this data. What happens to train vs test?

**3. Decision memo.** Write five lines to your VP: why CloudWave’s churn model will stay a GBT this quarter.

**4. Break the loop.** Comment out `opt.zero_grad()` and rerun 5 epochs. What happens to the loss? (It should explode or thrash.)
