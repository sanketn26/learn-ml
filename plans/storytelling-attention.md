# Plan: hold attention without changing the technical content

**Status:** diagnosis + rewrite playbook. No lesson rewrites in this PR.  
**Scope:** MkDocs lessons (`docs/`), exercise pages, hub pages. Not pipelines, tests, or datasets.  
**Trigger:** feedback that this is a *storytelling* problem, not a content problem. The ideas are sound. The pages do not hold attention.

---

## 1. Diagnosis

The course already has a latent plot:

> You join CloudWave → glue four systems into a Customer 360 → refuse a leaky feature → train a churn score → discover the label was a lie → rank 80 names for CS → pickle → nightly job → on-call → wrap the score as a bot tool.

On the page, that plot is almost never *told*. What the reader meets instead is a **highly consistent lesson template** with CloudWave as the sample path:

```
title
Course: …
Who this is for: Engineers who have [backend analog]
🎯 What you will be able to do
!!! think "Think of it like…"
## If you already write software     ← often restates the think box
ASCII picture of a mechanism
imports + code
!!! warning "Watch out"
!!! success "Ship / don’t ship"
When you can explain the week out loud, do the exercises
🤔 Reflection (always three questions)
🔗 Next week: [topic name]
```

That template is excellent *instructional design*. It is also the attention killer. After three weeks the reader can skip the first 40–80 lines, predict the emotional arc (analogy → picture → snippet → don’t-ship), and never enter a room at CloudWave.

**CloudWave is a dataset brand, not a company.** There are **zero named characters** in the lessons. Roles appear as nouns: “a PM,” “the CEO,” “the CFO,” “CS,” “a tired on-call engineer.” The reader is always generic **you**. Recurring objects are `subscriptions.csv`, `as_of=2024-06-01`, `FEATURE_COLS`, and ~6.4% churn — not a person who changed because last week’s model shipped.

The pedagogy promised in `CLAUDE.md` and `docs/ml/index.md` (`analogy → visual → math → predict → run → compare → explain`) is intact and loud. Story is not. Do not add more boxes. Do not add more topics. **Put people, time, and consequence around the content that already exists.**

---

## 2. What is not the problem

Do not “fix” these. They are the course’s strengths and the feedback did not ask for them to change.

- Analogy-first teaching, ASCII pictures, ship/don’t-ship rules.
- Engineer-to-engineer anti-hype voice (“the model is a guest,” “ReAct is not autonomy”).
- Laptop budget, no Jupyter, exercises as `starter.py`.
- Technical sequence of the job path (as-of features → gate → tonight’s CSV).
- Self-checks (Predict / Diagnose / Choose / Defend) — they already have more tension than most lessons; they are quiz-shaped, not story-shaped, which is fine for that page.

A rewrite that turns this into a cute novel, or that adds a fifth admonition type, misses the point.

---

## 3. The story that is already there (and where it breaks)

### Intended ML plot (readable from `docs/ml/index.md` + weeks 11, 15–17)

| Beat | Week | What should happen in the company |
|---|---|---|
| Glue language | 0–1 | First day. You cannot loop 160k rows. |
| Four systems, one customer | 2–3 | Finance’s MRR doubled. The CSV is a checkout, not the warehouse. |
| Chart for a decision | 4 | CFO wants one page. |
| Is the gap real? | 5 | Do not ship a plan change on 8/50 vs 12/60. |
| Feature contract | 6 | Scoring service cannot see the future. |
| Score + threshold | 7 | CS can call 80 people. Accuracy is a lie at 6.4% churn. |
| **Reversal** | 8 | Last week’s `is_churned` was a tenure detector. |
| Rank the queue | 11 | The product is an ordered list, not a yes/no. |
| Artifact | 15 | A pickle is not production. |
| Job | 16 | Gate, prod dir, tonight’s CSV. |
| **Payoff** | 17 | Pager. The Week 2 join, the Week 8 leak, the silent NaN — now in prod. Score becomes a tool. |

### Where the plot actually dies

1. **Weeks 0–6 are tooling chapters with CloudWave CSVs.** Week 0 does not put the reader at CloudWave until a late “load the CSV” section. Week 1’s CEO pulse is never answered as a decision.
2. **Weeks 9–10, 12–14, 18–20 are catalog units.** Regression, clustering, PCA, nets, CNN/RNN/Transformer pause or abandon the ship arc. Week 12 even admits the through-line does not have the problem: “CloudWave’s customer table has ~7 numeric fields.”
3. **Week 5 answers Week 4’s cliffhanger with the wrong product.** Week 4 closes: *“The CFO asks: ‘16% vs 20% churn — is that real?’”* Week 5 rolls out **Premium vs Standard**, which are not CloudWave plans (`free` / `starter` / `pro` / `enterprise`).
4. **Week 7 ships a label Week 8 calls a lie**, but Week 7 does not end on that reversal. The correction is a new chapter, not a consequence.
5. **Week 17 is the only media-res week**, and it *catalogues* the earlier bugs as a field guide instead of calling back to scenes the reader lived.
6. **Capstone leaves the company** for a generic coding-tool-use specialist. CloudWave’s Week 17 incidents become “a free, ground-truth dataset.”
7. **Framework tracks restart at “what is a chain.”** They share a logo, a golden file, and a refund motif. They do not continue the on-call shift.

### Scenario headings exist — and are rare

Only **seven** lessons have `## 🏢 Scenario`:

- ML: weeks 1, 2, 4, 5, 6, 9
- LangChain: week 3

The other ~30 weeks never even claim a scene. Several of the seven are schema tables or anti-pattern demos, not scenes (Week 2 “four systems, one customer”; Week 9 “don’t predict fake CLV”).

---

## 4. Systemic gaps (the attention problems)

### G1. Openings are syllabus cards, not cold opens

Almost every week starts with `Course:` / `Who this is for:` / a 🎯 list *before any trouble exists*. Week 1 even reprints the colored-box legend that already lives on the homepage.

The reader learns to skip the first screen.

**Evidence:** Week 0 opens “Python is not ‘the AI.’ It is the **glue**.” Week 9 opens “Anyone who has dragged a trendline in a spreadsheet.” Week 10 opens “Sorting Without Labels.” Contrast Week 11, which actually hooks: “CloudWave CS cannot call everyone. They can call **80**.”

### G2. No cast — attention has nowhere to attach

Zero named people. “A PM says” is a rhetorical device, not a character who will be wrong again next week. Framework tracks have Alice/Bob — in a **retail laptop shop**, not CloudWave.

Without a repeating human, every week is a new lecture. With five named roles, Week 17’s pager is *Ana’s*, and incident 1 is the Week 2 join she didn’t have a test for.

### G3. CloudWave is a filename, then a disclaimer

ML weeks 6, 9, 10, and 16 do not say “CloudWave” at all (grep). Framework weeks repeatedly wet-blanket the fiction:

> “Hypothetical CloudWave volume for this track: a few hundred tickets a day, not a vendor case study.”

That line (and its cousins in LC1, LC2, LC4, LC6, Crew1, Crew2, LG1) trains the reader *not to care*. One disclaimer on the track index is enough. Inside a week, write as if the company is real.

`docs/data.md` is honest about the split: “One company all the way through the **ML course**.” The homepage still claims “one company throughout” next to a file/grain/rows table.

### G4. Formula fatigue is advertised

`docs/ml/index.md` sells the cycle. Week 1 sells the box legend. Think-box and “If you already write software” often **say the same thing twice** (Week 6 leakage lecture is the type specimen). Admonitions are textbook callouts, not story beats.

Closings are ritual: “When you can explain the week out loud, do the exercises…” then three reflection questions then “Next week: [abstraction].”

### G5. Stakes are “you should know this because ML”

A week holds attention when *this week at CloudWave* needs the idea. Many weeks do not:

| Week | Why it feels like a unit from another course |
|---|---|
| 5 | Premium/Standard is a different product universe |
| 7 second half | Bias/variance on `sin` polynomials — leaves SaaS |
| 9 | Usage regression so the catalog has regression |
| 10 | Explicitly *not* the churn API |
| 12 | Table is too small for the curse of dimensionality |
| 14 | “You are learning nets so weeks 18–20 make sense” |
| 18–20 | Optional pictures; GBT refrain kills urgency |
| LC2 | CloudWave sells laptops and running shoes |
| LG1 | Spam / `"BUY NOW"` moderation, not the ticket bot |
| LG2 | KYC / limit-raise, never established |
| LG3 | `node1` / `node2` / `node3 exploded` — no company |
| Crew 2–4 | Changelog factory, a second product |
| Crew 3 | `assert seq.process == Process.sequential` |

### G6. Tension is postponed, then listed

- Exploding join: a `print` in Week 2, an incident in Week 17.
- Wrong label: shipped in Week 7, defined in Week 8, paged in Week 17.
- Scaler leak / silent NaN: taught as rules, then appear as “symptom boxes.”

Week 17 is the plot payoff compressed into one field guide. The reader did not *live* the incidents when they were taught, so the callbacks have nothing to hook.

### G7. Four tracks, one logo, three plots

| Track | Plot it actually tells |
|---|---|
| ML 0–17 | Latent churn-ship arc, told as a syllabus |
| LangChain 1–6 | Middleware chapters; week 7 finally assembles a bot |
| LangGraph | State machine; refund appears, then evaporates into `node3` |
| CrewAI | Staffing tutorial; index *says you probably don’t need this* |

The only readable cross-track sequence is: **the bot may classify (LC7); a human must approve money (LG4); resume must not charge twice (LG5).** CrewAI never enters that sequence. ML Week 17’s three incidents are never reused. `user_041906` is the best continuity object in the whole site and is unnamed.

### G8. Exercises continue the API, not the incident

Shared skeleton: What you are building → Predict before you run → Task 1–n → Success criteria → Debugging clues.

Predict-before-run is the only consistent tension, and it is a quiz. Week 7 exercise is “precision at a 100-call budget,” not “Priya can call 100 test users — send tonight’s list.” Week 17’s incident write-up is the exception, and it is a postmortem assignment, not a live debug.

### G9. Hub pages are a catalog, then a gate

- Homepage: four SKUs + dataset table. No “CloudWave is losing accounts and you just joined.”
- `docs/ml/index.md`: 21-row topic table. First sentence is a syllabus disclaimer.
- `getting-started.md`: readiness checklist and Docker before any ticket exists.
- `framework-tracks.md`: how to take the course before a problem exists.

Fine as *reference*. Fatal as *the first pages a learner reads*.

---

## 5. Heat map

Attention quality of the *story*, not of the teaching. “Best” still has no named cast.

### Strongest (keep the beat, add faces)

| Page | Why it almost works |
|---|---|
| ML Week 17 | Media res. Named failure modes with symptoms. |
| ML Week 8 | Reversal of last week. Correction is a story engine. |
| ML Week 11 | First lines are capacity and a queue. |
| ML Week 5 | A decision to make (wrong company, right shape). |
| ML Week 4 | A meeting brief (CFO, four questions). |
| LC Week 7 | “The one that can fail CI.” Golden file, `user_041906`. |
| LG Week 5 | Money + crash. Best title on the site. |
| LG Week 4 | Pause before a $50 refund. Three endings. |
| LC Week 3 | A customer actually asks a question (`$128.40`). |

### Weakest (template + no stake this week)

| Page | Main killer |
|---|---|
| ML Week 0 | Language spec. CloudWave is a CSV at the end. |
| ML Week 7 (second half) | Interview-unit bias/variance on sine waves. |
| ML Week 9–10 | Side quests in the middle of a ship path. |
| ML Week 12 | JPEG lecture + “this table is too small.” |
| ML Week 15 recap table | Full syllabus dump mid-climax. |
| ML Weeks 18–19 | Optional pictures; CloudWave is a tensor source. |
| LC Week 2 | Wrong product (Dell XPS, running shoes). |
| LC Week 6 | Dockerfile in a bot serial. |
| LG Week 1 | Spam moderation demo, then a disclaimer that it isn’t. |
| LG Week 3 | `node3 exploded`. |
| CrewAI Week 1 | Construct `Agent`, do not `kickoff()`. |
| CrewAI Week 3 | Assert a process flag. |
| `docs/ml/index.md` | Catalog, not a cold open. |

---

## 6. Rewrite playbook (same technical content)

These are the patterns. Apply them; do not invent new ML topics.

### P1. Cast card — five people, reused weekly

Add a short “who you work with” on `docs/ml/index.md` (and a one-line pointer on framework indexes). Suggested roster — names are placeholders, the *roles* are what matter:

| Person | Role | Recurring mistake / need |
|---|---|---|
| **you** | New ML engineer | Ship the list, then the job, then survive the pager |
| **Priya** | CS lead | 80 calls a week. Wants names, not AUC |
| **Marcus** | PM | Quotes charts, asks to “improve PC3,” wants both precision and recall |
| **Ana** | On-call / data | Will refuse promote if dummy wins. Owns the join test |
| **Helen** | CFO | One page, dollars, “is 16% vs 20% real?” |

Rules:

- Every “a PM says” becomes a line from Marcus.
- Every CS budget is Priya’s desk.
- Week 17’s pager is Ana’s; incident 1 is the Week 2 join.
- Do not add more characters. Five is the ceiling.

### P2. Cold open, syllabus second

First 8–12 lines of every week:

1. A Slack / ticket / standup line (or a number that is wrong).
2. One sentence of what is at risk *this week*.
3. Then the analogy.

Move `Who this is for`, 🎯, and “If you already write software” below the fold or into a collapsed details block. Stop restating the think box. Stop reprinting the box legend after Week 0.

**Before (Week 8 is close; make all weeks this shape):**

> About **6.4%** of customers ever cancel in this file. A model that predicts “nobody churns” is ~94% accurate and useless.

**After, with cast:**

> Priya called 80 names from last week’s list. None of them cancelled. Helen wants to know why we staffed a tenure detector. The column is `is_churned`.

Same technical payload (horizon labels, censoring, PR-AUC). Different first screen.

### P3. One CloudWave decision per week, paid off the same week

If you cannot finish the sentence “this week Priya/Marcus/Helen/Ana needs ___,” the week is a side quest. Either tie it to the ship arc or mark it **optional / skippable** in the nav so the plot can continue.

Concrete bindings (content already in the files):

| Week | Decision to put in the opening |
|---|---|
| 1 | Helen’s daily pulse: send median/p90, not the mean |
| 2 | Finance’s MRR doubled — collapsing the many-side is the scene, not a print |
| 3 | Ana will not train on a CSV someone emailed |
| 4 | Helen’s one-pager (already a Scenario — keep it, name her) |
| 5 | Use **starter vs pro** (or free vs paid) from the actual file, not Premium |
| 6 | Ana cannot compute the column at noon Tuesday |
| 7 | Priya has 80 slots. End the week on “we trained on ever-churned” |
| 8 | Priya’s list was a tenure detector (pay off 7) |
| 9 | Helen wants next-month usage for capacity, not fake CLV — or mark optional |
| 10 | Personas for Priya’s 80-call playbook — or mark optional |
| 11 | The product is Priya’s ordered queue (already the best stake line) |
| 12 | Marcus asked for a 2-D map; we will not staff PC3 — or mark optional |
| 13 | Bake-off for *tonight’s* list, not “ensembles as a topic” |
| 14 | Honest: GBT still wins on 7 columns; we learn nets for sequences later |
| 15 | Delete the recap table (it belongs on `index.md`). Ship night. |
| 16 | Ana refuses promote if dummy wins. Cliffhanger: you are on-call tomorrow |
| 17 | Call back: “this is the Week 2 join, now in prod” |

### P4. Plant incidents when the bug is taught

Week 17 should be a *reunion*, not a first meeting.

- Week 2 exploding MRR = “Helen’s dashboard doubled overnight” (same numbers).
- Week 6 scaler leak = Ana’s silent bad Tuesday.
- Week 8 wrong label = Priya’s empty week of calls.
- Week 17 then quotes those threads.

### P5. Closings are next-week *stakes*, not topic titles

Not: “PCA: JPEG for tabular data.”  
Instead: “Tuesday the pickle is in prod. Next week the job is allowed to write `artifacts/prod` — or not.”

Week 7 should end: “We just trained on ‘ever churned.’ Next week Priya finds out that is a tenure detector.”

Keep the exercise link. Drop or rotate the identical “When you can explain the week out loud…” sentence.

### P6. Admonitions as beats, once each

- `think` = one analogy, once.
- `warning` = what almost shipped this week.
- `success` = the Slack/email you actually send.
- Do not run think + engineer + software-table as three paraphrases.
- One “Hypothetical CloudWave, not a vendor case study” on the track index. Zero inside weeks.

### P7. Exercises continue the same ticket

Rewrite the *framing* of `docs/*/exercises/week-XX.md`, not the TODOs in `starter.py` (unless a TODO is “print AUC” when the story is “send Priya a list”).

- Week 0: Priya wants plan churn in Slack, no Pandas yet.
- Week 7: Priya can call 100 test users — send the list, not AUC.
- Week 16: Ana will refuse promote if dummy wins.
- Week 17 already almost does this — make 0–16 match.

### P8. Framework tracks: one incident, or drop the logo

Pick a single ticket and stop rotating worlds.

**Proposed through-line:** `CW-1847` — export timeout on ~150k rows (already in LC1 / LC4 / Crew2) → refund request (LC3 / LC7 / LG4) → resume double-charge (LG5) → optional changelog that dropped the `>200k` risk (Crew 2–4, if Crew stays). Same customer `user_041906`. Same on-call (Ana).

Must-fix world leaks (same mechanism, new strings):

| Week | Today | Same lesson, CloudWave strings |
|---|---|---|
| LC2 | Alice/Bob shop for Dell XPS / shoes | Two **tenants**; history = export still timing out vs API-key rotation |
| LG1 | `"BUY NOW"` spam moderation | Branch `docs` vs `refund_queue` (already the exercise graph) |
| LG2 | KYC / limit-raise | Email + on-call Slack on the export/refund ticket (already in the week as a flash) |
| LG3 | `node3 exploded` | Crash the week-4 refund graph after HTTP 200 |
| Crew 1–4 | Changelog *or* ticket analyst, never both | Researcher/writer/QA triage `CW-1847`, *or* stop prefixing CloudWave |

LC5–7 should be one CI story: the angry-billing golden case is `t2`/`t3` from `eval/golden_tickets.jsonl`, not a new `g1`/`g2`. Week 6 timeout is that handler hanging on `get_churn_score`. Week 7 is “the build went red in *your* repo.”

Connect to ML Week 17 by reusing incidents, not only `get_churn_score`.

### P9. Optional DL and capstone: stay in-company or drop the logo

- Weeks 18–20: “Marcus wants usage-*shape*, not totals — and it still loses to GBT” is a scene. Graduation is “you shipped churn and survived on-call,” not a three-pillar checklist.
- Capstone: fine-tune on CloudWave’s incident bot (continue Week 17) **or** stop saying CloudWave. Do not mine the pager as anonymous synthetic trajectories.

### P10. Hub pages: one cold open, then the catalog

`docs/index.md` and `docs/ml/index.md` should open on the company (churn, 80 calls, a nightly CSV, a pager) in ~5 lines, then the track picker. Keep the dataset table; it is not the hook.

`getting-started.md` can stay a gate. Link it second, not first.

---

## 7. Phased implementation

Do this in PRs that a reviewer can read. Do not rewrite 37 weeks in one diff.

### Phase 0 — bible (small, unblocks everything)

- Add `docs/cloudwave.md` (or a section on `docs/ml/index.md`): cast, company one-pager, the job-path plot, the one framework incident (`CW-1847`), what CloudWave is *not* (no laptops, no KYC, no Premium plan).
- Do **not** add it to a learner’s critical path as another lecture. Authors use it; lessons show it.

### Phase 1 — openings and closings of the job path (highest attention ROI)

ML weeks **4, 5, 7, 8, 11, 15, 16, 17** plus `docs/ml/index.md`.

- Cold open + named cast.
- Week 5: replace Premium/Standard with real plan names from `subscriptions.csv`.
- Week 7 close → Week 8 open as one reversal.
- Week 15: delete the in-body syllabus table.
- Week 17: rewrite incidents as callbacks.

This phase alone should change the “I bounced” feedback. The middle catalog weeks can wait.

### Phase 2 — plant the bugs when they are taught

ML weeks **0–3, 6**. First-day scene, exploding-join as Finance Slack, as-of as Ana refusing an emailed CSV, leakage as a Tuesday that already happened.

### Phase 3 — catalog weeks: bind or mark optional

ML **9, 10, 12, 13, 14, 18–20**. Nav labels may say “side quest” / “optional picture.” Do not pretend PCA is this week’s production crisis.

### Phase 4 — exercises framing

Exercise markdown for the Phase 1 weeks first, then 0–6. `starter.py` TODOs stay unless the success criterion is still “print a metric” when the story is “send a list.”

### Phase 5 — framework tracks as one incident

LC7 + LG4 + LG5 first (already the best sequence). Then LC1–6 string fixes. Then LG1–3 (especially LG1 spam and LG3 `node3`). CrewAI last: either join `CW-1847` or drop the brand.

### Phase 6 — capstone continuity (optional)

Only after Week 17 has a cast and callbacks. Otherwise the capstone will keep mining anonymous incidents.

---

## 8. Definition of done (how we know attention improved)

A week is “story-complete” when all of these are true:

1. The first screen is a CloudWave beat (person, ticket, or wrong number), not a 🎯 list.
2. A named person needs this week’s idea *this week*.
3. The closing names a consequence, not a topic.
4. If Week 17 will page you for this bug, the bug already happened on-page in the teaching week.
5. No world leak (Premium, laptops, KYC, Reddit spam, `node3`) unless the week is explicitly marked off-plot.
6. The exercise is the next shift on the same ticket.

Author test: **read only the first 12 and last 8 lines of weeks 4→5→7→8→11→16→17 in order.** If that reads as a serial, the spine works. If it still reads as a syllabus, it does not.

Learner test (qualitative, matches the original feedback): “I wanted to see what happened next” vs “I knew the next box was Watch out.”

---

## 9. Explicitly out of scope

- New algorithms, new datasets, GPU, Jupyter.
- Changing the pedagogy order (analogy still comes early — just *after* the cold open).
- Turning Ship/don’t-ship into a narrative essay. Keep it as a decision rule; make the decision *someone’s*.
- Adding a mascot, comic, or long fiction between code blocks.
- Publishing this plan on the MkDocs site (`plans/` is author-facing).

---

## 10. Suggested first PR after this plan lands

**Title:** *Story bible + cold opens for the job-path spine (weeks 4, 5, 7, 8, 11, 16, 17).*

Why that slice: it is the latent plot, it already contains the best lines on the site, and Week 5’s Premium/Standard break is a one-hour factual fix with outsized story payoff.
)
