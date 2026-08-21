# Week 5 — “Is This Real, or Just Noise?”

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who ship A/B tests and get asked “but is it significant?” You do not need a stats degree.

We will **not** memorize a zoo of tests. We will make one decision carefully, then keep a flowchart for later.

---

## 🎯 What you will be able to do

- Translate a p-value into a sentence a PM cannot misuse
- Run the actual “8 / 50 vs 12 / 60” launch question — and see it fail to reject
- Draw a confidence interval as “a range of plausible true rates”
- Know which test matches your column types
- Refuse to ship on p &lt; 0.05 alone

!!! think "Think of it like… a code review, or a courtroom."

    The **null hypothesis** is the boring default: “these two plans churn the same; the difference is luck.” You do *not* prove the new plan works. You ask: *if they were the same, how often would luck produce a gap this big?* That frequency is the p-value. Innocent until proven guilty. High bar to convict.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from lib.course_data import find_data_dir

DATA = find_data_dir()
```


## If you already write software

A p-value is a flaky-test statistic, not a trophy.

You already know this feeling: a test failed once on CI. Is the build broken, or did the suite sneeze? You do not ship on one red run. You ask: *if the code were fine, how often would this fail anyway?*

That frequency is the p-value.

```
Null hypothesis     “these two plans churn the same; the gap is luck”
p-value             how often a no-difference world produces a gap this big
0.05 threshold      a house style, not a law of nature
significant         “weird enough that luck is an awkward explanation”
NOT significant     “we do not know yet”  ← not “they are equal”
```

### What a p-value is not

- Not “the probability we are wrong”
- Not “the probability Premium is worse”
- Not “how big the effect is” (that is the effect size / the interval)
- Not permission to ship

`8/50` vs `12/60` *looks* like Premium wins. With that few customers, coin-flips produce a 4-point gap all the time. The PM sees 16% vs 20%. You see a sample size.

### Picture the courtroom

Innocent until proven guilty. The null is the defendant. You need a high bar to convict. Failing to convict is not the same as proving innocence — it means “go get more data, or pick a bigger effect to care about.”

## 🏢 Scenario — should we roll out Premium?

Early data:

| Plan | Churned | Customers | Rate |
|---|---|---|---|
| Premium | 8 | 50 | **16%** |
| Standard | 12 | 60 | **20%** |

A PM sees “Premium is better.” An engineer asks: **with this few customers, how often would a 4-point gap appear by coin-flip?**

!!! engineer "Engineer mental model"

    A p-value is *not* “the probability we are wrong.” It is not “the probability Premium is worse.” It is: **how often a world with no real difference produces a result this spicy.** Same idea as “how often would this flaky test fail on a green build?”

## Visual: luck can look like a win

We will fake 10,000 worlds where both plans truly churn at 18%. In each world, draw 50 + 60 customers. Plot the Premium − Standard gap. Then mark the gap we actually saw (−4 points).

```python
rng = np.random.default_rng(42)
true_rate = 0.18
n_prem, n_std = 50, 60
observed_gap = 8 / 50 - 12 / 60  # -0.04

sim_gaps = rng.binomial(n_prem, true_rate, 10_000) / n_prem - rng.binomial(n_std, true_rate, 10_000) / n_std

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(sim_gaps, bins=40, color="#93c5fd", edgecolor="white")
ax.axvline(observed_gap, color="#b91c1c", lw=2, label=f"observed gap {observed_gap:.0%}")
ax.axvline(0, color="#334155", ls="--", label="no difference")
ax.set_title("If both plans were 18% churn, 4-point gaps happen all the time")
ax.set_xlabel("Premium rate − Standard rate")
ax.legend()
plt.tight_layout()
plt.show()

p_two_sided = (np.abs(sim_gaps) >= abs(observed_gap)).mean()
print(f"Share of fake worlds with a gap at least this big: {p_two_sided:.2f}")
print("That is a p-value, built with a for-loop in your head instead of a formula.")
```

## The same answer, with a library test

Chi-squared (or Fisher’s exact, for tiny counts) is the grown-up version of the histogram above.

!!! math "Math, translated"

    A p-value of 0.03 would mean: *in a no-difference world, about 3 in 100 reruns look this extreme.* It does **not** mean “there is a 3% chance Premium is a bad idea.” The 8/50 vs 12/60 launch story is *not* that world — its p is large, so luck still explains a 4-point gap at this sample size.

```python
table = np.array([[8, 42],   # premium: churned, retained
                  [12, 48]])  # standard
chi2, p, dof, expected = stats.chi2_contingency(table)
print("Chi-squared p-value on the 8/50 vs 12/60 story:", round(p, 3))
print("Expected counts if plans were equal:\n", expected.round(1))
print("\nVerdict: p is large. We do NOT have enough evidence to declare Premium better.")
print("Ship decision: keep collecting data. Do not rewrite billing based on 110 customers.")
```

## Now the full CloudWave table

Same question, real `subscriptions.csv`. More customers → the same 4-point gap would be a much bigger deal.

!!! tip "Visual cue — which test?"

    **Category vs category** (plan × churned) → chi-squared.

    **Number vs 2 groups** (MRR for churned vs not) → t-test (or Mann-Whitney if the histogram is a whale-tail).

    **Number vs 3+ groups** (usage by region) → ANOVA, then look at the picture before you trust the p.

```python
subs = pd.read_csv(DATA / "subscriptions.csv")

ct = pd.crosstab(subs["plan_type"], subs["is_churned"])
print("Counts:\n", ct)
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-squared p-value across all plans: {p:.2e}")

rates = subs.groupby("plan_type")["is_churned"].agg(["mean", "count"])
# Exact binomial (Clopper–Pearson-style) interval — not a Wilson interval
cis = []
for plan, row in rates.iterrows():
    lo, hi = stats.binom.interval(0.95, int(row["count"]), row["mean"])
    cis.append((plan, row["mean"], lo / row["count"], hi / row["count"], row["count"]))
ci_df = pd.DataFrame(cis, columns=["plan", "rate", "lo", "hi", "n"]).set_index("plan")
print("\n95% range of plausible churn rates:")
print(ci_df.round(3))

fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(ci_df))
ax.errorbar(ci_df["rate"], y,
            xerr=[ci_df["rate"] - ci_df["lo"], ci_df["hi"] - ci_df["rate"]],
            fmt="o", color="#1d4ed8", capsize=4)
ax.set_yticks(y, ci_df.index)
ax.set_xlabel("churn rate")
ax.set_title("Confidence interval = plausible range for the true rate, not a vote of confidence")
plt.tight_layout()
plt.show()
```

## A number vs two groups — do churners pay less?

T-test asks: “is the difference in average MRR bigger than the usual jitter in averages?”

```python
churned = subs.loc[subs["is_churned"] == 1, "mrr"]
kept = subs.loc[subs["is_churned"] == 0, "mrr"]
t, p = stats.ttest_ind(churned, kept, equal_var=False)
print(f"Mean MRR churned={churned.mean():.1f}  kept={kept.mean():.1f}")
print(f"Welch t-test p={p:.3g}")

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(kept.clip(upper=200), bins=40, alpha=0.6, label="kept", color="#22c55e")
ax.hist(churned.clip(upper=200), bins=40, alpha=0.7, label="churned", color="#ef4444")
ax.set_title("MRR distributions (clipped at $200) — look before you t-test")
ax.set_xlabel("MRR")
ax.legend()
plt.tight_layout()
plt.show()

print("Free users have MRR = 0 and churn more. The t-test may just be rediscovering the free plan.")
```

!!! warning "Watch out"

    - **p-hacking:** 20 slices of the data will produce one “p < 0.05” by accident. Pre-register the question, or treat extra slices as exploration.

    - **Significance ≠ importance:** with ~49k rows, a 0.2% churn gap can be “significant” and still not worth an engineering quarter.

    - **CI overlap** is a sloppy shortcut. Look at the interval on the *difference*, or just look at dollars.


!!! success "Ship / don’t ship"

    Ship when (1) the interval on the lift is mostly above your *business* threshold, (2) you have looked at the chart, (3) a second slice (another month, another region) rhymes. p < 0.05 is a filter, not a launch button.

!!! warning "A ranker is not a lever"

    Later weeks will rank who looks like they will churn. That is **prediction**. “If we increase usage, they will stay” is **causation**. Plan × churn in this file is **observational** — people chose their plan. A chi-squared p-value does not make it an experiment. Causation still needs a randomized experiment, not a feature-importance plot (Week 11).


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-05.md). Starter: `python exercises/ml/week-05/starter.py` from the repo root.

## 🤔 Reflection

1. Explain a p-value to a PM in one sentence without the word “significant.”
2. Why did 8/50 vs 12/60 fail, while the full table’s plan comparison did not?
3. You ran 12 ad-hoc tests on one Friday. How many “wins” do you expect by luck at α = 0.05?

## 🔗 Next week

Feature engineering — turning Customer 360 columns into the **API contract** of a model, without leaking the future into the past.
