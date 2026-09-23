---
description: CloudWave story bible — the cast, the company, the job-path plot, and the framework-track incident, for authors keeping the course's continuity straight.
hide:
  - toc
---

# CloudWave story bible

This page is for authors, not a required lesson. It is the single source of truth for names, numbers, and the incident chain, so weeks 0–17 and the framework tracks stay one continuous story instead of restarting each time. Lessons should **show** this; they should not require a learner to **read** this first.

## The cast (five names, no more)

| Person | Role | Recurring need / mistake |
|---|---|---|
| **you** | New ML engineer | Ship the list, then the job, then survive the pager |
| **Priya** | CS lead | Can call **80** names a week. Wants names, not AUC |
| **Marcus** | PM | Quotes charts, wants both precision and recall, asks for things that sound good and cost accuracy |
| **Ana** | On-call / data engineer | Owns the join test and the feature contract; refuses to promote a model that loses to the dummy |
| **Helen** | CFO | Wants one page, in dollars; asks whether a gap is real before anyone touches pricing |

Use these names instead of "a PM," "the CFO," "CS," or "a tired on-call engineer." Do not add a sixth — five is the ceiling so a reader can hold the whole cast in their head by week 2.

## The company, honestly

CloudWave is a fictional B2B SaaS company invented for this course. Its only real artifacts are the CSVs in `data/`. There is no CloudWave product screenshot, pricing page, or support macro beyond what a lesson writes inline.

**Real, from `data/subscriptions.csv` (48,991 rows, as of 2024-11-30):**

- Plans: `free` (24,400 customers, 9.0% churn), `starter` (14,622, 3.8%), `pro` (7,420, 3.6%), `enterprise` (2,549, 3.8%)
- Lifetime churn across all customers: **6.4%** (3,122 of 48,991)
- Columns: `user_id, plan_type, mrr, signup_date, churn_date, is_churned, tenure_days`

**What CloudWave is not** — do not let a lesson drift into:

- **Premium / Standard** plan tiers (Week 5 used to; the real tiers are `free/starter/pro/enterprise` above — starter and pro churn at nearly identical rates, so don't invent a 16%-vs-20% gap between real tiers; keep illustrative small-sample scenarios clearly hypothetical, not a real plan comparison)
- A laptop or running-shoe retailer (a LangChain memory demo used to)
- A bank running KYC or credit-limit checks (a LangGraph demo used to)
- A Reddit-moderation or generic spam filter (a LangGraph demo used to)
- A second, unrelated changelog-generation product (a CrewAI demo used to) — unless a week is explicitly marked optional/off-plot

One line of "hypothetical CloudWave, not a vendor case study" belongs on each track's index page. Inside a week, write as if the company is real.

## The ML job-path plot (weeks 0–17)

> You join CloudWave → glue four systems into a Customer 360 → refuse a leaky feature → train a churn score → discover the label was a lie → rank 80 names for CS → pickle → nightly job → on-call → wrap the score as a bot tool.

| Beat | Week | Scene |
|---|---|---|
| First day, glue language | 0–1 | You cannot loop 160k rows by hand |
| Four systems, one customer | 2–3 | Helen's MRR dashboard doubles overnight — a join fanned out |
| Chart for a decision | 4 | Helen wants one page: is churn getting worse, which plan leaks, do engaged customers stay |
| Is the gap real? | 5 | Marcus wants to declare an early rollout a win off 50 and 60 customers |
| Feature contract | 6 | Ana can't compute a feature by noon Tuesday — it needs data from the future |
| Score + threshold | 7 | Priya has 80 calling slots; the model you trained learned "ever churned," not "will churn soon" |
| **Reversal** | 8 | Ana blocks the handoff: that's a tenure detector, not a 30-day churn score |
| Rank the queue | 11 | The product is Priya's ordered list of 80, not a yes/no |
| Artifact | 15 | A pickle on your laptop is not production |
| Job | 16 | Ana won't let anything ship that loses to the dummy; the pipeline gets a gate |
| **Payoff** | 17 | You're on-call. Three incidents you already lived come back: the join that doubled Helen's MRR, the leak Ana caught, and a silent NaN that slipped past both |
| **Capstone: ship** | after 17 | Quarter-end. Helen wants "it runs every week" in the board pack; Priya wants her 80 every Monday; Ana wants a holiday. You ship the job. Two weeks later (2024-06-15) Priya says Monday's list "looks like strangers" — one seeded upstream defect (dollars→cents, an events fan-out, or a stalled usage extract) that passed every contract check |

## The one framework-track incident: `CW-1847`

A single support ticket carries the LangChain, LangGraph, and (optionally) CrewAI tracks, instead of each week inventing a new fictional company:

1. **`CW-1847` opens** — a customer's data export times out around 150k rows (LangChain weeks 1, 4; CrewAI week 2 background)
2. **Refund requested** — the customer asks for a refund while the export is broken (LangChain weeks 3, 7; LangGraph week 4 — a human must approve the money)
3. **Resume, don't double-charge** — LangGraph week 3's `issue_credit` node crashes mid-retry on the `CW-1847` thread, establishing that resume is at-least-once, not exactly-once. Week 5 then shows the idempotency-key fix for that failure mode on a **different** write — a payment capture, `charge(invoice_id=...)`, on a different customer's invoice — because a refund payout and a charge capture are different operations and forcing them into one demo would misrepresent one as the other. Both need the pattern; only one is dramatized end-to-end.
4. **Optional: the changelog that missed it** — if the CrewAI track stays in a course run, its researcher/writer/QA crew triages `CW-1847` instead of an unrelated changelog product

`user_041906` is `CW-1847`'s customer, and the same ID used in ML Week 17 and the LangChain golden-ticket fixture `t3` ("is `user_041906` about to cancel?"). `t2` (the prompt-injection refund attempt) carries **no customer ID** — do not describe it as `user_041906`'s ticket. LangGraph week 5's `charge()` demo deliberately uses a different, unnamed invoice, not `user_041906`'s — see point 3. Same on-call engineer throughout: **Ana**.

`CW-1847` is a proposed continuity device for the framework tracks — it is not yet wired into every lesson. Treat any week that doesn't yet reference it as a to-do, not a contradiction.

## Golden tickets (`eval/golden_tickets.jsonl`)

- `t1` — balance/billing question, no tool call
- `t2` — prompt-injection attempt asking for a refund; must refuse
- `t3` — "is `user_041906` about to cancel?"; expects `get_churn_score`
- `t4`, `t5` — export/API-key questions, no tool call

`t2` and `t3` are different fixtures with different expected behavior (refusal vs. churn-score lookup) — do not treat them as interchangeable "angry billing" cases.

## What stays out of this page

- Invented measured results (a specific AUC, a specific "N of Priya's calls converted") that no code actually computed. Keep narrative facts (names, ticket IDs, dates) separate from computed output (metrics, test results) — see the author ledger below.
- Technical corrections that belong in code, not in a story. If a lesson's math or code has a defect, fix the code; don't paper over it with narrative.

## Author ledger

Keep this list current as lessons change. It is the fast way to check "does this number match what's used elsewhere."

| Fact | Value | Source |
|---|---|---|
| Total customers | 48,991 | `data/subscriptions.csv` |
| Lifetime churn rate | 6.4% (3,122) | `data/subscriptions.csv` |
| Plan churn rates | free 9.0%, starter 3.8%, pro 3.6%, enterprise 3.8% | `data/subscriptions.csv` |
| Priya's weekly call budget | 80 | Week 11, Week 17 |
| Week 7 exercise test-set budget | 100 | `exercises/ml/week-07/starter.py` |
| Framework incident ID | `CW-1847` | proposed, this page |
| Continuity customer | `user_041906` | Week 17, LC golden tickets |
| On-call engineer | Ana | Week 16, Week 17 |
| Job-path capstone ship date / incident night | 2024-06-01 / 2024-06-15 | `docs/ml/capstone-ship.md` |
| Churners at risk on 2024-06-01, 90-day horizon | 110 of 43,947 (0.25%) | `snapshot_split` |
| Desk list worth (90-day horizon) | ~1 churner per 80 calls, ~5× a random 80 | `docs/ml/capstone-ship.md` |
| Scenario-bank budgets | Helen $3,000 discounts (20% × 3 mo); Marcus 50 pitches; Ana 30 senior slots; onboarding 40 sessions (first 45 days) | `capstone_ship/briefs.py` |
