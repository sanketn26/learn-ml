---
description: Use hypothesis testing and p-values to judge whether an A/B test result is real or noise, with confidence intervals as a gut check.
---

# Week 5 — “Is This Real, or Just Noise?”

Marcus turned on a new onboarding checklist for a slice of new signups two weeks ago. He wants to tell Helen it cut churn by a third before Monday's meeting. The slice was 50 customers on the old flow and 60 on the new one.

??? note "Course details"

    **Course:** Applied ML Foundations for SaaS Analytics
    **Who this is for:** Engineers who ship A/B tests and get asked "but is it significant?" You do not need a stats degree.

We will **not** memorize a zoo of tests. We will make one decision carefully, then keep a flowchart for later.

---

## 🎯 What you will be able to do

- Translate a p-value into a sentence a PM cannot misuse
- Run the actual “8 / 50 vs 12 / 60” launch question — and see it fail to reject
- Draw a confidence interval as “a range of plausible true rates”
- Know which test matches your column types
- Refuse to ship on p &lt; 0.05 alone

!!! think "Think of it like… a code review, or a courtroom."

    The **null hypothesis** is the boring default: "the checklist changed nothing; the difference is luck." You do *not* prove Marcus's checklist works. You ask: *if it changed nothing, how often would luck produce a gap this big?* That frequency is the p-value. Innocent until proven guilty. High bar to convict.

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
Null hypothesis     “the checklist changed nothing; the gap is luck”
p-value             how often a no-difference world produces a gap this big
0.05 threshold      a house style, not a law of nature
significant         “weird enough that luck is an awkward explanation”
NOT significant     “we do not know yet”  ← not “they are equal”
```

### What a p-value is not

- Not “the probability we are wrong”
- Not “the probability the checklist doesn’t work”
- Not “how big the effect is” (that is the effect size / the interval)
- Not permission to ship

Marcus remembers `8/50` vs `12/60` as the checklist winning — a 4-point gap, in his telling. (Spoiler for the scenario below: he has the columns backwards.) Even if the direction were right, with that few customers coin-flips produce a 4-point gap all the time. Where he sees a win, you see a sample size.

### Picture the courtroom

Innocent until proven guilty. The null is the defendant. You need a high bar to convict. Failing to convict is not the same as proving innocence — it means “go get more data, or pick a bigger effect to care about.”

## 🏢 Scenario — did the checklist actually work?

Two weeks of early data, both groups of new signups:

| Group | Churned | Customers | Rate |
|---|---|---|---|
| Old flow | 8 | 50 | **16%** |
| New checklist | 12 | 60 | **20%** |

Wait — that's backwards from what Marcus is claiming. He remembered the headline number, not the table. That is itself the week's first lesson: **check the data before you check the p-value.** An engineer asks a second question anyway: **with this few customers, how often would a 4-point gap this size appear by coin-flip, in either direction?**

These 50/60 customers are a small, hypothetical early slice — not CloudWave's full `starter`/`pro` populations, which run within half a point of each other at n in the thousands (see the full-table test below). Don't read this toy example as "CloudWave's real plans differ by 4 points."

!!! engineer "Engineer mental model"

    A p-value is *not* “the probability we are wrong.” It is not “the probability the checklist doesn’t work.” It is: **how often a world with no real difference produces a result this spicy.** Same idea as “how often would this flaky test fail on a green build?”

## Visual: luck can look like a win

We will fake 10,000 worlds where both groups truly churn at 18%. In each world, draw 50 + 60 customers. Plot the old-flow − checklist gap. Then mark the gap we actually saw (−4 points).

```python
rng = np.random.default_rng(42)
true_rate = 0.18
n_old, n_new = 50, 60
observed_gap = 8 / 50 - 12 / 60  # -0.04

sim_gaps = rng.binomial(n_old, true_rate, 10_000) / n_old - rng.binomial(n_new, true_rate, 10_000) / n_new

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(sim_gaps, bins=40, color="#93c5fd", edgecolor="white")
ax.axvline(observed_gap, color="#b91c1c", lw=2, label=f"observed gap {observed_gap:.0%}")
ax.axvline(0, color="#334155", ls="--", label="no difference")
ax.set_title("If both groups were 18% churn, 4-point gaps happen all the time")
ax.set_xlabel("old-flow rate − checklist rate")
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

    A p-value of 0.03 would mean: *in a no-difference world, about 3 in 100 reruns look this extreme.* It does **not** mean “there is a 3% chance the checklist is a bad idea.” The 8/50 vs 12/60 story is *not* that world — its p is large, so luck still explains a 4-point gap at this sample size.

```python
table = np.array([[8, 42],   # old flow: churned, retained
                  [12, 48]])  # checklist
chi2, p, dof, expected = stats.chi2_contingency(table)
print("Chi-squared p-value on the 8/50 vs 12/60 story:", round(p, 3))
print("Expected counts if the groups were equal:\n", expected.round(1))
print("\nVerdict: p is large. We do NOT have enough evidence to tell Marcus the checklist worked.")
print("Ship decision: keep collecting data. Do not tell Helen this is proven on 110 customers.")
```

## Now the full CloudWave table

Different question — is churn different *by plan*, not by checklist — but the same method, run on the real `subscriptions.csv` instead of a 110-customer toy. More customers → a 4-point gap would be a much bigger deal, if one existed.

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

counts = subs.groupby("plan_type")["is_churned"].agg(["sum", "count"])
# Exact (Clopper–Pearson) interval on the true binomial proportion —
# stats.binom.interval gives a range of *outcomes*, not a confidence interval
# on the parameter. proportion_ci(method="exact") is the Clopper–Pearson CI.
cis = []
for plan, row in counts.iterrows():
    k, n = int(row["sum"]), int(row["count"])
    result = stats.binomtest(k, n)
    lo, hi = result.proportion_ci(method="exact")
    cis.append((plan, k / n, lo, hi, n))
ci_df = pd.DataFrame(cis, columns=["plan", "rate", "lo", "hi", "n"]).set_index("plan")
print("\n95% Clopper–Pearson range of plausible churn rates:")
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

1. Explain a p-value to Marcus in one sentence without the word “significant.”
2. Why did 8/50 vs 12/60 fail, while the full table’s plan comparison did not?
3. You ran 12 ad-hoc tests on one Friday. How many “wins” do you expect by luck at α = 0.05?

## Before you leave

Try one [self-check](self-checks.md#week-5-signal-vs-noise) (Predict / Diagnose / Choose / Defend). Write the answer before you open the block.

## 🔗 Next week

You tell Marcus to wait for more data. Ana has a sharper problem: she's building the features for a churn model, and one of the columns you'd love to use — next month's usage — doesn't exist yet at scoring time. [Week 6](week-06.md) is the **API contract** a feature has to honor, without leaking the future into the past.
