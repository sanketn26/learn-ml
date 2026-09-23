---
description: A bank of CloudWave business briefs — discount targeting, expansion ranking, support deflection, onboarding activation — that judge the same churn score by different definitions of success.
---

# Capstone scenario bank — same model, five definitions of "shipped"

Priya's desk is one customer of the churn score. By the end of the job-path capstone, three more people have asked for it. Helen wants to spend a discount budget. Marcus wants an upsell list that doesn't pitch anyone halfway out the door. Ana wants the angriest tickets from the riskiest customers in front of senior agents. And Priya's onboarding team wants to know which brand-new accounts to call first. Nobody is asking for a new model. They are each asking a different question of the one you shipped.

A brief is that question written down: who is eligible, how many — or how many dollars — someone can act on, how the list is ordered, and what success means in *their* units. The model, the artifact, and the code path stay the same. Only the brief changes.

!!! think "Think of it like… one API, five clients with different SLAs."

    You don't fork a service because the mobile app and the billing batch job need different timeouts. You keep one endpoint and let each client state its contract. A churn score is the endpoint. Each brief is a client contract: a filter, a sort, a limit, and an acceptance test.

## The picture

```
                          ┌─ retention-desk        Priya   80 calls        → precision vs base rate
                          ├─ discount-targeting    Helen   $3,000 budget   → break-even save rate
 one artifact ─► scores ──┼─ expansion-ranking     Marcus  50 pitches      → churners pitched (lower is better)
                          ├─ support-deflection    Ana     30 senior slots → recall among ticket-holders
                          └─ onboarding-activation Priya   40 sessions     → precision inside new accounts
                                 │
                   select(frame, scores, brief) → judge(picked, frame, y, brief)
```

```python
from pathlib import Path

from capstone_ship.briefs import BRIEFS, judge, select, threshold_for
from pipelines.contract import load_artifact
from pipelines.features import FEATURE_COLS
from pipelines.split import snapshot_split
from pipelines.train import train

_, _, test_df, y_test = snapshot_split("2024-06-01", horizon_days=90)
meta = train("2024-06-01", Path("artifacts/briefs"), n=8000, horizon_days=90)
scores = load_artifact(Path("artifacts/briefs") / meta["model_version"])["pipeline"].predict_proba(test_df[FEATURE_COLS])[:, 1]

for brief in BRIEFS.values():
    picked = select(test_df, scores, brief)
    print(f"{brief.key:<22} n={len(picked):<3} threshold={threshold_for(picked):.4f}  {judge(picked, test_df, y_test, brief)}")
```

One loop, five lists, five thresholds. `capstone_ship/briefs.py` holds each brief as data; adding a sixth is a dataclass, not a pipeline.

## The briefs

### Retention desk — Priya (the capstone's default)

- **Constraint:** 80 calls a week.
- **List:** every at-risk customer, highest score first.
- **Shipped means:** hits / 80, stated next to the base rate — what a random 80 would reach.
- **Write-up question:** is the lift worth a CS salary's worth of calls?

### Discount targeting — Helen

- **Constraint:** $3,000 of retention discounts: 20% off for three months. Free accounts can't be discounted.
- **List:** paid accounts in risk order, taken until the next discount would overrun the budget.
- **Shipped means:** the **break-even save rate** — the share of reached churners who must stay *because of the offer* for it to pay for itself, valuing a saved customer at a year of MRR (a stated assumption, in `VALUE_MONTHS`).
- **Write-up question:** above 1.0 the discount can't pay for itself even if it saves everyone it reaches. What precision would the list need to get under 0.3?

!!! math "Math, translated — why this list isn't sorted by MRR"

    A proportional discount on an account costs `k × mrr` and protects `mrr`. Return per discount dollar is `score × mrr / (k × mrr) = score / k` — **the account's size cancels out**. Sorting by `score × mrr` looks like "protect the most revenue" and actually spends the budget on a handful of large accounts that rarely leave. Sort by risk; let the budget decide how many.

### Expansion ranking — Marcus

- **Constraint:** sales can pitch an upgrade to 50 accounts (free, starter, or pro).
- **List:** engaged accounts, ordered by usage × (1 − churn score) — busy *and* safe.
- **Shipped means:** churners pitched, next to a usage-only list of the same size. Here the churn score is a **guardrail**, not the target: success is a *low* number.
- **Write-up question:** if the guardrail version and the usage-only version pitch the same number of churners, is the churn score earning its place in this list?

### Support deflection — Ana

- **Constraint:** 30 senior-agent slots a week.
- **List:** customers with at least one support ticket, highest score first.
- **Shipped means:** recall of churners among ticket-holders — how many of the at-risk people who are *already talking to us* reach a senior agent — next to the ticket-holders' own base rate.
- **Write-up question:** the eligible pool is small. Does narrowing the population to people already in the support queue make the same score more or less useful than it is on the desk list?

### Onboarding activation — Priya's onboarding team

- **Constraint:** 40 guided sessions a week, for accounts in their first 45 days.
- **List:** new accounts only, highest score first.
- **Shipped means:** precision *inside the new-account slice*, against that slice's base rate — not the whole population's.
- **Write-up question:** what did this model mostly learn, and how much of it is left once every candidate is new?

!!! warning "Watch out — a brief can remove the model's best feature"

    The churn score leans hardest on `tenure_so_far`: new accounts churn. The onboarding brief keeps *only* new accounts, so inside its pool the strongest signal is nearly constant. A model can have excellent overall AUC and nothing useful to say inside a slice. Always judge a brief against **its own slice's** base rate, and check whether the list beats a random pick from that slice before anyone staffs sessions against it.

## Attaching a brief to a capstone

Every brief takes a frame, a score per row, and backtest labels. Any capstone that produces a churn score can be judged by any brief:

- **Job-path capstone** — [step 4](capstone-ship.md#4-the-threshold-is-a-headcount) takes a `brief` argument. Swap `RETENTION_DESK` for any entry in `BRIEFS`; the threshold in `metrics.json` and the write-up change, nothing else does.
- **Any later model** — score the same `test_df` with it and pass the scores to `select`. A model that wins on AUC and loses on the brief your stakeholder cares about has not won.

## Ship / don't ship

!!! success "Ship / don't ship"

    **Ship** a list when you can state its value in the stakeholder's units — calls, dollars, pitches, sessions — against the baseline *their* population would get by chance. **Don't ship** a list judged only by the model's global metrics, a budget spent by the wrong sort key, or a slice nobody checked against its own base rate.

## ✍️ Exercise

Task 9 of the [job-path capstone exercises](exercises/capstone-ship.md): run two briefs through your step 4 and write a ship / don't-ship for each.
