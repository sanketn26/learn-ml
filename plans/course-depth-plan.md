# Plan — descriptions, staged hints, and three more capstones

Three observations, three workstreams. Ordered by risk: cheapest and most
mechanical first, so the expensive work lands on a clean base.

---

## Workstream 1 — de-leak the `description:` front-matter

**Problem.** `description:` is the meta description: it renders in search
results and social cards, *before* the learner opens the page. Several
descriptions state the finding the exercise exists to produce. This directly
defeats the `## Predict before you run` section, which asks the learner to
commit to a guess the description already answered.

**Rule to apply.** A description names the *task and the dataset*, never the
*result, the bug's name, or the direction of an effect*. Verbs like "see X
drop", "why X skews Y", "forbidden", "leaky", "mistake", "commented out" are
the tell.

**Confirmed leaks** (the pass will re-scan all 84 pages; these are the ones
already identified):

| Page | Leaks | Fix direction |
|---|---|---|
| `ml/exercises/week-06.md` | "honest vs a leaked feature scaler" | "Compare two scaler fitting strategies across a train/test split" |
| `ml/exercises/week-09.md` | "forbidden leaky CLV target ... inflates R-squared" | "Build an alternative CLV target and audit its R-squared" |
| `ml/exercises/week-10.md` | "to see feature scale dominate distance" | "Run K-Means on raw MRR and usage, then on scaled" |
| `ml/exercises/week-13.md` | "correct a common soft-voting vs stacking naming mistake" | "Name the ensemble method in a worked example" |
| `ml/exercises/week-14.md` | "training loop with zero_grad commented out" | "Diagnose a PyTorch training loop that runs but does not learn" |
| `ml/exercises/week-15.md` | "to catch data leakage" | "Compare shuffled vs time-split AUC" |
| `ml/exercises/week-18.md` | "compare it against a dense-flatten baseline" (mild) | keep task, drop implied verdict |
| `ml/exercises/week-19.md` | "test whether the model relies on order" (mild) | acceptable — it is the question, not the answer |
| `ml/exercises/week-20.md` | "to see accuracy drop" | "Ablate positional encoding and measure the change" |
| `ml/week-10.md` | "why unscaled features skew customer segments" | "how feature scale interacts with distance" |

**Scope.** Front-matter only. Lesson bodies are *supposed* to teach the
finding — they are not leaks. No nav or CSS change.

**Verify.** `mkdocs build --strict`, plus a grep for the tell-words across
front-matter to confirm none survive.

---

## Workstream 2 — staged hints

**Problem.** `pymdownx.details` is already enabled and `???` collapsibles
already work, but only 2 of 38 exercise pages use one. Meanwhile every ML
exercise page carries a flat `## Debugging clues` section that is *always
visible* — a spoiler sitting above the work rather than a hint the learner
chooses to open.

**Scheme — three stages, each its own collapsible, all closed by default:**

```markdown
??? tip "Hint 1 — a nudge"
    Which of the two splits can see the future? Nothing about the code yet.

??? tip "Hint 2 — the approach"
    Fit the scaler inside the split, not before it. Two lines move.

??? example "Hint 3 — most of the code"
    ```python
    scaler.fit(X_train)          # not X
    X_train_s = scaler.transform(X_train)
    ```
```

Stage 1 reframes the question. Stage 2 names the approach without code.
Stage 3 shows near-complete code but stops short of the write-up/judgment
the task asks for. A learner opens only as far as they need, and the cost of
peeking is visible to them.

**Migration of `## Debugging clues`.** Its existing bullets are mostly good
stage-1/stage-2 material. They get folded into the staged blocks per task
rather than deleted — but they stop being visible by default. Where a bullet
is a genuine *environment* gotcha rather than a hint ("Do not `import
unsloth` on the laptop", "`plan_type` must be `str`"), it stays visible: that
is setup, not a spoiler.

**Rollout order.** ML exercises 00–17 (the required path) first, then the
capstones, then 18–20, then the framework tracks. Batched so each commit is
reviewable.

**Verify.** `mkdocs build --strict`; spot-render `mkdocs serve` to confirm
the collapsibles nest correctly inside numbered tasks (indentation inside
`???` blocks is the usual failure).

---

## Workstream 3 — three more capstones

**Problem.** The one existing capstone is the *fine-tuning* one: optional,
GPU-requiring, and positioned past the optional DL weeks. A learner who
completes the required job path (weeks 0–17) finishes with **no integrative
project at all**. The capstone that exists serves the smallest audience.

Each new capstone follows the established shape: `docs/ml/capstone-<slug>.md`
lesson + `docs/ml/exercises/capstone-<slug>.md` + `exercises/ml/capstone-<slug>/starter.py`
+ a `tests/test_capstone_<slug>.py` gate + a `solutions/ml/capstone-<slug>/`
entry. All three are CPU-only, unlike the existing one.

### 3a. Job-path capstone — "Ship the churn score" (weeks 0–17) — highest value

The missing keystone. Closes the required track end to end, reusing
`pipelines/` rather than re-teaching it:

1. As-of feature build at a chosen date (`build_features`, week 3/6 grain rules)
2. Label with an explicit horizon (week 8 censoring)
3. Train + beat a dummy on a *time* split (weeks 7, 15)
4. Choose a threshold against a stated capacity budget, not 0.5 (weeks 11, 15)
5. Wrap in `validate`/`predict` with a contract test (weeks 6, 15)
6. Promote-gate it and run a scheduled score (week 16)
7. Injected incident: one seeded defect the learner must find via postmortem (week 17)

Step 7 is what makes it a capstone rather than a long exercise — it is graded
on the diagnosis, not just a metric.

### 3b. Agent/framework capstone — closes LangChain + LangGraph

A CloudWave support agent that must survive its own operations, not just work
once: a golden-file CI gate (LC week 5/7), a keyword firewall and a real
"I don't know" path (LC week 4/7), checkpointed resume (LG week 3), human
approval before any write (LG week 4), and an idempotency key so a resume
never double-acts (LG week 5). Runs on `FakeListLLM` — no API key, CPU-only,
consistent with the tracks it closes.

### 3c. Deep-learning capstone (weeks 18–20) — CPU-feasible

Gives the optional DL arc a landing place without breaking the no-GPU rule.
A sequence task over `user_events` — predict churn from an event *sequence*
rather than an aggregated row — implemented three ways (CNN, RNN,
transformer) against the **week-13 GBT as the baseline to beat**. The
intended finding is that the GBT often wins on this data shape and scale;
the deliverable is an honest verdict, which is exactly the lesson.

### 3d. Business scenarios (your addition)

Rather than a fourth capstone page, a **scenario bank** that re-skins the
capstones: same harness, different business framing and success metric
(pricing/discount targeting, expansion-revenue ranking, support-deflection,
onboarding-activation). Each scenario is a short brief — stakeholder, budget
constraint, what "shipped" means — that changes the threshold and the
write-up without changing the code path. Cheap to add, and it is the part
that "opens up minds": the same model, judged by four different business
definitions of success.

**Open question for you (3d):** scenario bank as *one* page of briefs
attachable to any capstone, or *one brief per capstone* baked in? I lean
one page of briefs — more reuse, less duplication.

---

## Sequencing

1. **W1 descriptions** — mechanical, low risk, unblocks nothing but fixes a live leak. One commit.
2. **W2 hints** — ML 00–17 first, batched commits.
3. **W3a job-path capstone** — the biggest real gap.
4. **W3d scenario bank** — cheap once 3a exists, and 3a is the first consumer.
5. **W3b agent capstone**, then **W3c DL capstone**.

Each workstream is independently shippable; nothing later depends on
anything earlier except 3d on 3a.

## Cross-cutting constraints (from CLAUDE.md)

- Pedagogy order preserved: analogy → picture → code → foot-gun → ship/don't-ship
- CPU-only for all three new capstones; only the existing Phase 3 breaks that
- CloudWave continuity — new capstones reuse the cast and the CW-1847 thread;
  `docs/cloudwave.md` is the story bible and may need new entries
- Every new page added to `mkdocs.yml` nav
- `mkdocs build --strict` and `pytest tests/` green at each commit
