---
description: The deep-learning capstone — CNN, RNN, and transformer encoders over CloudWave event sequences, benchmarked on CPU against the Week-13 GBT with a control and several seeds.
---

# Capstone — Does the Order of Events Beat the Row?

Marcus has read a blog post: *every churn model should be a transformer over the event stream now.* Helen's question is shorter: *does it make Priya's list better, and does it need a GPU?* You have the churn score from the job path, the three architectures from Weeks 18–20, and a CPU. This capstone answers Marcus with a benchmark instead of an opinion — and the honest answer is allowed to be "no."

??? note "Course details"

    **Closes:** the optional deep-learning weeks (18–20), with Week 13's GBT as the bar.
    **Runs on:** CPU, the main venv. The full bake-off — six models, three seeds, two backtest dates — takes a minute or two.

---

## 🎯 What you will be able to do

- Turn an event log into leak-free, padded sequences on the same backtest split the job path uses
- Build CNN, RNN, and transformer encoders that respect padding and see the same static row as the GBT
- Run a bake-off with **two controls**, **several seeds**, and **more than one date**, and read a gap against its spread
- Separate what the *events* add from what their *order* adds
- Test whether a model uses *order* by scrambling it
- Write a verdict that says "the simpler model wins" when that's what the numbers say

!!! think "Think of it like… benchmarking a new query planner."

    You don't ship a new planner because it's newer. You run it against the current one on the same queries, you add a control that isolates the one thing that changed, and you run it enough times that a difference is bigger than the noise. The GBT is the current planner. The sequence is the one thing that changed.

## The picture

```
 one at-risk customer, as of 2024-06-01 (median: 5 events; max kept: 16)

 tokens   [PAD PAD PAD … PAD  login  page_view  login]   ← left-padded: position -1 is the latest event
 recency  [ 0   0   0  …  0   0.92     0.90     0.65 ]   ← log-days before as_of
 static   [mrr, tenure_so_far, log_usage, …, plan]       ← the same row the GBT sees

            ┌─ GBT (week 13)          static only           ← the bar
            ├─ MLP                    static only           ← control 1: no events at all
 scores ◄───┼─ bag of events + static same events, no order ← control 2: types + recency, mean-pooled
            ├─ CNN         + static   sliding detector      (week 18)
            ├─ GRU         + static   walking clipboard     (week 19)
            └─ transformer + static   everything-looks-at-everything (week 20)
```

Adding a sequence adds three things at once: *which* events happened, *how recently*, and the *order* they came in — plus a bigger model to fit them. Comparing an encoder with the static MLP measures all of that together. The bag control sees the same events and the same recency through the same step embedding, but pools them with a mean, so it cannot tell login-then-cancel-page from the reverse. That splits the gain in two:

```
 events added = bag − static MLP          which events, how recent
 order  added = encoder − bag             the sequence itself
```

```python
from capstone_sequence.data import EVENT_TYPES, NONE, bakeoff_data

train, test = bakeoff_data()
events_per_row = test.mask.sum(axis=1)
print(f"test: {len(test.y):,} customers, {int(test.y.sum())} churn within 30 days")
print(f"events per customer: median {int(sorted(events_per_row)[len(events_per_row) // 2])}, "
      f"max {events_per_row.max()}, no events at all {(test.tokens[:, -1] == NONE).sum():,}")
print("latest event, first 3 customers:", [EVENT_TYPES[t - 1] if t != NONE else "NONE" for t in test.tokens[:3, -1]])
```

A median of five events is not much of a sequence. Hold that thought.

## The data, and the two rules that keep it honest

`capstone_sequence.data.bakeoff_data` builds on the same backtest as the job path: train on the snapshot 30 days before `as_of`, test on the full `as_of` population — never downsampled. Two representation rules matter:

- **Left padding.** The latest event is always at position `-1`, so a GRU reads its answer there without packing, and a CNN or transformer ignores padding through `mask`.
- **Silent customers get a `NONE` token.** 205 test customers have no events before `as_of`. An all-padding row makes an attention mask fully blank and produces NaNs — and "no events" is information anyway.

## Three encoders, one contract

Each encoder is `(tokens, recency, mask, static) -> logit`, built from a shared step embedding and a shared head:

| Encoder | Week | Pools the sequence by | Padding handled by |
|---|---|---|---|
| CNN | 18 | loudest hit of a 3-wide detector | `masked_fill(-inf)` before the max |
| GRU | 19 | the state after the latest event | left padding: read position `-1` |
| Transformer | 20 | mean over attended events | `src_key_padding_mask=~mask`, masked mean |

## Read the gap against the spread

`bakeoff` trains every model over several seeds with a class-weighted loss (positives are ~2% of customers; ~7% of training rows once negatives are downsampled to 8,000) and reports mean and standard deviation of AUC and PR-AUC, lift over the dummy, and hits in the top 80 (ranked by score, ties broken by `user_id`, like every list in the course). `bakeoff_dates` runs the whole thing on two backtest dates. Three readings matter:

1. **Bag vs static MLP** — what the events added: which ones, how recent.
2. **Encoder vs bag** — what *order* added, on top of the same events.
3. **Best model vs GBT** — whether any of this is worth changing what ships.

Then one experiment for *why*: scramble each test customer's real events into a random order, tokens and recency together. The bag can't notice — it never saw order. A model that uses order should get worse.

!!! warning "Watch out — an event name is not a label"

    The event log has a `cancel` type, logged on the day a customer churns. It looks like the perfect feature. Check where it can appear: a customer at risk on `as_of` has, by definition, not churned yet — so no at-risk sequence ever contains `cancel`. If one does, your `as_of` cut is broken and the model is reading the answer key. The label comes from `churn_date` in `subscriptions.csv`, what finance booked; an event name is what an instrumentation engineer typed, and it only means what the cut lets it mean.

!!! warning "Watch out — a gap smaller than the seed spread is not a result, and seeds are not dates"

    Top-80 hits are a handful at most on this data, and they move between seeds; AUC moves by a few thousandths. A model that "wins" by less than its own spread hasn't won. But a small seed spread only says the result is stable *on this month's customers*. Re-run on a second backtest date: if the ranking of models flips between dates, the seed spread was hiding the real uncertainty. Report the spread next to every number, on more than one date, or don't report the number.

??? success "The reference run — open after your own bake-off"

    Three seeds, 20 epochs, 8,000 training rows, the full test snapshot, 30-day label, on two backtest dates:

    | Model | AUC 06-01 (± sd) | PR-AUC 06-01 | AUC 09-01 (± sd) | PR-AUC 09-01 |
    |---|---|---|---|---|
    | GBT (week 13) | 0.744 | 0.051 | 0.773 | 0.063 |
    | MLP, static only | 0.753 ± 0.002 | 0.056 | 0.766 ± 0.001 | 0.063 |
    | Bag of events + static | 0.758 ± 0.002 | 0.052 | 0.768 ± 0.001 | 0.058 |
    | CNN + static | 0.756 ± 0.004 | 0.054 | 0.761 ± 0.003 | 0.055 |
    | GRU + static | 0.756 ± 0.001 | 0.052 | 0.768 ± 0.001 | 0.057 |
    | Transformer + static | 0.748 ± 0.006 | 0.052 | 0.770 ± 0.002 | 0.059 |

    Read it in the three steps. **Events** (bag − static): +0.005 and +0.002 AUC — a little, on both dates — while PR-AUC *falls* on both (0.056 → 0.052, 0.063 → 0.058): the extra inputs help the ranking in the middle of the list and hurt it at the top, which is where Priya's 80 live. **Order** (best encoder − bag): −0.002 on June 1, +0.002 on September 1 — the same size as the seed spread, pointing different ways on different dates. That is not a finding. Scrambling order costs the GRU about 0.004 AUC, so it *reads* order a little; reading it isn't the same as it being worth anything. **The bar**: the static MLP beats the GBT on June 1 and loses to it on September 1. A seed spread of ±0.002 made the June result look solid; the second date says the ranking of the simple models is a coin that flips by month. The churn signal in this file is *how much* a customer is still doing, which the static row already carries. Marcus's transformer doesn't ship, the GBT stays, and none of it needed a GPU to find out.

## Ship / don't ship

!!! success "Ship / don't ship"

    **Ship** an architecture change when it beats the model in production by more than its seed spread, on more than one backtest date, *and* beats a control that isolates what it added — the bag, if it claims order — on the same brief. **Don't ship** a model because its architecture is newer, because one seed or one month looked good, or because a benchmark compared it with a control that differs in three things at once.

## ✍️ Exercise

[Deep-learning capstone exercises](exercises/capstone-sequence.md) — write the three encoders in `exercises/ml/capstone-sequence/starter.py`, run the bake-off, scramble the order, and write the verdict for Marcus.

## 🤔 Reflection

1. What would the event log need — longer histories, different event types, a different question — before order could plausibly matter?
2. The MLP beat the GBT on June 1 and lost on September 1. How many dates would you want before replacing the GBT — and what else would you check first?
3. Which scenario-bank brief is most sensitive to a small PR-AUC difference, and which barely notices it?
