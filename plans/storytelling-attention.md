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

That template supports consistent instruction. Its repetition is a plausible contributor to the reported attention problem, not a demonstrated cause. After three weeks the reader can skip the first 40–80 lines, predict the emotional arc (analogy → picture → snippet → don’t-ship), and never enter a room at CloudWave.

**CloudWave is a dataset brand, not a company.** There is **no established recurring CloudWave cast** in the ML lessons; named example users do appear in framework lessons. Roles appear as nouns: “a PM,” “the CEO,” “the CFO,” “CS,” “a tired on-call engineer.” The reader is always generic **you**. Recurring objects are `subscriptions.csv`, `as_of=2024-06-01`, `FEATURE_COLS`, and ~6.4% churn — not a person who changed because last week’s model shipped.

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

1. **Weeks 0–6 are tooling chapters with CloudWave CSVs.** Week 0 mentions CloudWave in its opening objectives, but does not establish a first-day company scene. Week 1’s CEO pulse is never answered as a decision.
2. **Weeks 9–10, 12–14, 18–20 are catalog units.** Regression, clustering, PCA, nets, CNN/RNN/Transformer pause or abandon the ship arc. Week 12 even admits the through-line does not have the problem: “CloudWave’s customer table has ~7 numeric fields.”
3. **Week 5 answers Week 4’s cliffhanger with the wrong product.** Week 4 closes: *“The CFO asks: ‘16% vs 20% churn — is that real?’”* Week 5 rolls out **Premium vs Standard**, which are not CloudWave plans (`free` / `starter` / `pro` / `enterprise`).
4. **Week 7 teaches a lifetime label Week 8 replaces.** Its code comment, warning, and final link already explicitly foreshadow the correction. The missing piece is a human decision and payoff, not a missing technical transition. Week 7 uses `tenure_so_far`, not the forbidden lifetime `tenure_days`.
5. **Week 17 is the only media-res week**, and it *catalogues* the earlier bugs as a field guide instead of calling back to scenes the reader lived.
6. **Capstone changes the learner’s role** to coding-tool-use specialist, but explicitly reuses CloudWave’s Week 17 incidents. Preserve that existing continuity and strengthen the handoff; it is not a wholly unrelated project.
7. **Framework tracks restart at “what is a chain.”** They share a logo, a golden file, and a refund motif. They do not continue the on-call shift.

### Scenario headings exist — and are rare

Only **seven** lessons have `## 🏢 Scenario`:

- ML: weeks 1, 2, 4, 5, 6, 9
- LangChain: week 3

The other 30 weeks never even claim a scene. Several of the seven are schema tables or anti-pattern demos, not scenes (Week 2 “four systems, one customer”; Week 9 “don’t predict fake CLV”).

---

## 4. Systemic gaps (the attention problems)

### G1. Openings are syllabus cards, not cold opens

Almost every week starts with `Course:` / `Who this is for:` / a 🎯 list *before any trouble exists*. Week 1 even reprints the colored-box legend that already lives on the homepage.

The reader learns to skip the first screen.

**Evidence:** Week 0 opens “Python is not ‘the AI.’ It is the **glue**.” Week 9 opens “Anyone who has dragged a trendline in a spreadsheet.” Week 10 opens “Sorting Without Labels.” Contrast Week 11, which actually hooks: “CloudWave CS cannot call everyone. They can call **80**.”

### G2. No cast — attention has nowhere to attach

No recurring CloudWave cast. “A PM says” is a rhetorical device, not a character who will be wrong again next week. Framework tracks have Alice/Bob — in a **retail laptop shop**, not CloudWave.

Without a repeating human, every week is a new lecture. With five named roles, Week 17’s pager is *Ana’s*, and incident 1 is the Week 2 join she didn’t have a test for.

### G3. CloudWave is a filename, then a disclaimer

ML weeks 6, 9, 10, and 16 do not say “CloudWave” at all (grep). Framework weeks repeatedly wet-blanket the fiction:

> “Hypothetical CloudWave volume for this track: a few hundred tickets a day, not a vendor case study.”

That line (and its cousins in LC1, LC2, LC4, LC6, Crew1, Crew2, LG1) trains the reader *not to care*. Consolidate repeated fictional-company boilerplate on the track index. Keep local concept-demo and production-limit qualifications; write scenes consistently within the fictional company.

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

> Preserve the CrewAI index’s explicit reason to decline extra orchestration. Optionality is an instructional strength; a story should make the tradeoff concrete, not hide it.

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
| CrewAI | Staffing tutorial; index says “This track is optional ceremony on top of LangChain week 7.” |

The only readable cross-track sequence is: **the bot may classify (LC7); a human must approve money (LG4); resume must not charge twice (LG5).** CrewAI never enters that sequence. ML Week 17’s incidents are not a sustained framework-track story; the capstone does explicitly reuse them. `user_041906` is the best continuity object in the whole site and is unnamed.

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

Move `Who this is for`, 🎯, and “If you already write software” below the fold or into a `??? note "Who this is for"` collapsed block — `pymdownx.details` is already enabled in `mkdocs.yml` (line 56) and renders as a themed, click-to-expand admonition; raw `<details>` would work via `md_in_html` but loses Material's styling. No page currently uses this extension, so this is free capacity, not a new dependency. Stop restating the think box. Stop reprinting the box legend after Week 0.

**Before (Week 8 is close; make all weeks this shape):**

> About **6.4%** of customers ever cancel in this file. A model that predicts “nobody churns” is ~94% accurate and useless.

**After, with cast:**

> Priya asks for 80 names likely to cancel in the next 30 days. Last week’s notebook learned who ever cancelled. Ana stops the handoff: where is the 30-day answer key?

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
| 8 | Replace the lifetime answer key before Priya uses the queue (pay off 7) |
| 9 | Helen asks how well billing and event counts explain usage; the current target is same-snapshot `total_usage`, not next-month forecasting |
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
- Week 8 wrong label = Ana blocks the proposed queue until the 30-day label is defined. Do not invent observed call outcomes.
- Week 2 missing values and Week 6 feature contract = plant the fill-value mismatch that becomes Week 17’s silent-NaN incident. The scaler leak is useful context, but is not one of Week 17’s three incidents.
- Week 17 then quotes those threads.

### P5. Closings are next-week *stakes*, not topic titles

Not: “PCA: JPEG for tabular data.”  
Instead: “Tuesday the pickle is in prod. Next week the job is allowed to write `artifacts/prod` — or not.”

Week 7 should end: “We just trained on ‘ever churned.’ Priya needs ‘in the next 30 days.’ Next week we fix the answer key before handing over the queue.”

Keep the exercise link. Drop or rotate the identical “When you can explain the week out loud…” sentence.

### P6. Admonitions as beats, once each

- `think` = one analogy, once.
- `warning` = what almost shipped this week.
- `success` = the Slack/email you actually send.
- Do not run think + engineer + software-table as three paraphrases.
- Consolidate “Hypothetical CloudWave, not a vendor case study” on the track index. Retain locally necessary demo limitations (see §11).

### P7. Exercises continue the same ticket

Rewrite the *framing* of `docs/*/exercises/week-XX.md`, not the TODOs in `starter.py`. Record any task change as a separate implementation follow-up.

- Week 0: Priya wants plan churn in Slack, no Pandas yet.
- Week 7: Priya can call 100 test users — send the list, not AUC.
- Week 16: Ana will refuse promote if dummy wins.
- Week 17 already almost does this — make 0–16 match.

### P8. Framework tracks: connect the incidents

Pick a single ticket and stop rotating worlds.

**Proposed through-line:** `CW-1847` — export timeout on ~150k rows (already in LC1 / LC4 / Crew2) → refund request (LC3 / LC7 / LG4) → resume double-charge (LG5) → optional changelog that dropped the `>200k` risk (Crew 2–4, if Crew stays). Same customer `user_041906`. Same on-call (Ana).

Must-fix world leaks (same mechanism, new strings):

| Week | Today | Same lesson, CloudWave strings |
|---|---|---|
| LC2 | Alice/Bob shop for Dell XPS / shoes | Two **tenants**; history = export still timing out vs API-key rotation |
| LG1 | `"BUY NOW"` spam moderation | Branch `docs` vs `refund_queue` (already the exercise graph) |
| LG2 | KYC / limit-raise | Email + on-call Slack on the export/refund ticket (already in the week as a flash) |
| LG3 | `node3 exploded` | Name the failing step in a ticket workflow; retain the minimal graph and foreshadow the later write/replay example |
| Crew 1–4 | Changelog *or* ticket analyst, never both | Researcher/writer/QA triage `CW-1847`, *or* stop prefixing CloudWave |

LC5–7 should share a CI story, with an explicit fixture mapping: `t2` is injection/refund refusal; `t3` is the churn-score query. They are not interchangeable angry-billing cases. LC5’s `g1`/`g2` use a different demo schema; keep that distinction unless a separately scoped code change aligns the evaluators. Week 6 timeout is that handler hanging on `get_churn_score`. Week 7 is “the build went red in *your* repo.”

Connect to ML Week 17 by reusing incidents, not only `get_churn_score`.

### P9. Optional DL and capstone: make the role transition explicit

- Weeks 18–20: “Marcus wants usage-*shape*, not totals — what evidence would justify a net?” is a scene. Do not invent a measured loss to GBT. Graduation is “you shipped churn and survived on-call,” not a three-pillar checklist.
- Capstone: continue the existing Week 17 incident connection with Ana commissioning a coding assistant. Preserve its read/suggest tool contract and synthetic ground truth. Replacing it with a support bot is a separate curriculum/code change.

### P10a. Formatting affordances already enabled, currently unused

`mkdocs.yml`'s `markdown_extensions` block licenses more than the five admonition types in play:

- `pymdownx.details` (`??? note` / `???+ note` for expanded-by-default) — zero uses in `docs/`. This is the mechanism for P2's below-the-fold syllabus material, and could also collapse Week 15's recap table (§P5) instead of deleting information some readers still want.
- `pymdownx.tabbed` — zero uses in `docs/`. Could hold "software background" framing as a tab instead of a full paragraph duplicating the think-box (the G4 redundancy), so a reader who doesn't need the analogy restated can skip it without the author deleting it for everyone.
- `attr_list` — available for adding IDs/classes to headings or elements (e.g. a distinct visual treatment for the cold-open paragraph vs. body text) without new CSS work, since `docs/stylesheets/extra.css` already exists to target such classes.

None of this requires a new dependency or CSS framework — it is unused capability already declared in the build config. Verified by reading `mkdocs.yml` lines 47–69 and grepping `docs/` for `<details>`/`???` usage (zero hits both).

### P10b. Code block length is not currently a formatting problem

Checked: the longest code blocks in `docs/ml/` run 51–54 lines (weeks 3, 4, 14, 18, 20) — long but not "wall of code." No block currently needs splitting for formatting reasons alone; don't add `pymdownx.tabbed`-style code-block splitting as a P10a task unless a specific rewrite makes a block longer.

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

This phase tests whether concrete stakes improve the “I bounced” feedback; improvement is not established until learners try it. The middle catalog weeks can wait.

### Phase 2 — plant the bugs when they are taught

ML weeks **0–3, 6**. First-day scene, exploding-join as Finance Slack, as-of as Ana refusing an emailed CSV, leakage as a Tuesday that already happened.

### Phase 3 — catalog weeks: bind or mark optional

ML **9, 10, 12, 13, 14, 18–20**. Nav labels may say “side quest” / “optional picture.” Do not pretend PCA is this week’s production crisis.

### Phase 4 — exercises framing

Exercise markdown for the Phase 1 weeks first, then 0–6. `starter.py` TODOs and measurable success criteria stay; task changes require a separately scoped follow-up.

### Phase 5 — framework tracks as one incident

LC7 + LG4 + LG5 first (already the best sequence). Then LC1–6 string fixes. Then LG1–3 (especially LG1 spam and LG3 `node3`). CrewAI last: either join `CW-1847` or drop the brand.

### Phase 6 — capstone continuity (optional)

Strengthen the existing incident handoff after Week 17 has a cast and callbacks; preserve the coding-specialist objective.

---

## 8. Definition of done (how we know attention improved)

A week passes the proposed editorial checklist when these hold (subject to the usability and evidence safeguards in §11):

1. The first screen is a CloudWave beat (person, ticket, or wrong number), not a 🎯 list.
2. A person or concrete operational decision needs this week’s idea *this week*.
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

Why that slice: it is the latent plot and already contains strong decision points. Pilot weeks 4–5 and 7–8 first; review learner feedback before expanding to the remaining spine. Week 15 remains in Phase 1 as a subsequent small change. Do not estimate the Week 5 rename as a simple string fix until its statistical framing and exercise are reviewed.


---

## 11. Verification and implementation safeguards (2026-09-17)

**Verdict:** the repository supports the diagnosis of inconsistent story continuity and repeated framing. It does not establish that repetition causes drop-off, that every lesson needs named dialogue, or that a serial story will improve learning. Treat the rewrite as a testable editorial hypothesis. This audit checked source text and fixture data, not every runnable example or learner behavior.

### Verified evidence and corrections

| Finding | Evidence / qualification |
|---|---|
| 37 weekly lessons; seven Scenario headings | `docs/{ml,langchain,langgraph,crewai}/week-*.md`: 21 + 7 + 5 + 4. Seven literal `## 🏢 Scenario` matches, exactly the list in §3. A heading count is not a measure of engagement. |
| Section-order flow is identical across sampled weeks | Weeks 2, 6, 10, 12 all run 🎯 → "If you already write software" → [optional Scenario] → topic sections → ✍️ Exercise → 🤔 Reflection → 🔗 Next week, confirming §1's skeleton diagnosis empirically. `mkdocs.yml` nav order is a plain sequential list — not itself a flow defect. |
| Four ML lessons omit the company name | Weeks 6, 9, 10, 16, confirmed by literal search. They still use shared data/contracts; missing branding does not by itself mean missing continuity. |
| Actual subscription plans | `data/subscriptions.csv`: 48,991 rows; `free` 24,400, `starter` 14,622, `pro` 7,420, `enterprise` 2,549. The 8/50 and 12/60 illustration is not a computed sample of those plans. |
| Week 7 already sets up Week 8 | [Week 7](../docs/ml/week-07.md): lifetime-label code comment, warning, and closing. `numeric` includes `tenure_so_far`; do not rewrite history to say its model used lifetime `tenure_days`. |
| Week 17 has three specific incidents | [Week 17](../docs/ml/week-17.md): exploding join, forbidden lifetime features, divergent missing-value handling. Plant all three, not just the scaler warning. |
| Framework ticket is proposed continuity | `CW-1847` is an author proposal, not an existing shared fixture. LC3’s balance tool uses `user_0001`/`user_0002`; its $128.40 example is not already a refund for `user_041906`. |
| Capstone already reuses incidents | [Capstone Phase 2](../docs/ml/capstone.md) explicitly connects Week 17 to `capstone/scenarios.py`. Retain this and the optional GPU exception. |

### Missing guardrails to apply before rewriting

1. **Keep narrative facts separate from measured results.** Maintain a small author ledger of customer ID, ticket ID, plan, amount, snapshot date, horizon, budget, and evidence file. Distinguish invented dialogue from computed output. An empty call list or zero later cancellations does not alone demonstrate a broken model, particularly after an intervention. Keep the ~6.4% lifetime prevalence separate from the much rarer 30-day target in Week 8; 80 production calls and the Week 7 exercise’s 100 test-set users are different settings.
2. **Week 5 needs more than plan-name replacement.** Keep 8/50 vs 12/60 explicitly illustrative if those counts stay. A comparison of self-selected `starter` and `pro` users is observational, not a randomized launch test. Preserve the existing “A ranker is not a lever” warning and avoid claiming that plan choice caused churn differences.
3. **Track an existing technical defect separately.** Week 5 labels `stats.binom.interval(0.95, n, observed_rate) / n` a Clopper–Pearson confidence interval. It is a range of binomial outcomes under a supplied probability, not that parameter-confidence procedure. Use `stats.binomtest(k, n).proportion_ci(method="exact")` in a focused technical follow-up, with actual churn counts. See the official [binomial distribution API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html) and [proportion confidence interval API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). This contradicts a blanket “technical content needs no correction”; record it without silently expanding this editorial change.
4. **Do not promise an experiment the code does not run.** Week 9 predicts snapshot `total_usage`; “next-month capacity forecast” would require a future target and time-aware evaluation. Likewise, “GBT still wins” is a curriculum recommendation unless the particular comparison was actually run. LG5’s in-memory `CHARGES` demo calls a function twice in one process; it does not verify durability across a process restart. Preserve that limitation when writing a crash scene.
5. **A refund is not a charge.** The proposed framework serial must explain why LG4 approves a refund but LG5 demonstrates capture/charge idempotency. Keep them as related examples or scope a coordinated code/exercise change; a string rename cannot make them the same financial operation. Preserve human approval and the read-only ML/LC bot boundary. Session-key isolation in LC2 must not be presented as implemented tenant authorization.
6. **Keep lessons independently usable.** Add a one-sentence recap and an ordinary prerequisite link for readers arriving from search or directly into a framework track. Keep objectives, accessible headings, setup instructions, and concept-vs-integration labels easy to find. Consolidate repeated fictional-company boilerplate, but keep local limitations where a standalone snippet could otherwise be mistaken for production evidence. Rendered first screens, not raw line counts including YAML, determine whether a hook is visible.
7. **Avoid replacing one rigid template with another.** A concrete operational question, failed assertion, or anomalous number can open a week without dialogue. Names are optional when they distract from the mechanism. Keep one short scene; retain the analogy, predict/run/compare/explain cycle, and useful recap material. Move or shorten the Week 15 recap only after checking that its information remains discoverable.
8. **Optionality is a curriculum decision.** Weeks 0–17 are currently required in `CLAUDE.md`, `README.md`, and the ML index. Before marking 9/10/12/14 skippable, map downstream prerequisites and reconcile navigation, indexes, exercises, and self-checks. Prefer a brief story bridge first. Do not conceal the existing optional status of DL/CrewAI to manufacture urgency.
9. **Respect the framing-only boundary.** Keep starter TODOs, fixtures, expected outputs, algorithms, and success criteria unchanged in the editorial PR. “Send the list” must still require the existing precision/threshold/ablation work. Any proposed fixture/schema, task, or capstone change becomes a separately scoped follow-up, with matching lesson/exercise/solution updates and relevant checks.

### Additions to the rollout and acceptance checks

- **Phase 0:** record the continuity ledger and prerequisite map. Resolve the proposed week-7/8 outcome, Week 9 forecast wording, refund/charge transition, and `t2`/`t3` mapping before authoring scenes. Keep the story bible author-facing unless learners need a short cast pointer.
- **Phase 1 pilot:** compare original and revised weeks 4–5 and 7–8 with representative working engineers. Ask what decision they were solving, what they expected next, and whether they can still explain the statistical/label caveat. Record confusion and exercise completion as well as preference; a stronger desire to continue is insufficient if comprehension drops. Set a success criterion before reviewing feedback; treat a small qualitative pilot as directional, not causal proof.
- **Hub coverage:** assign `docs/index.md` to Phase 1 and framework indexes to Phase 5; §P10 previously had no explicit implementation phase. Preserve easy access to readiness/setup guidance. Align the ML index’s “start at Think of it like” reading instructions with the new opening order.
- **Per editorial PR:** review a local MkDocs preview at desktop and narrow widths; check headings, collapsed blocks, code visibility, prerequisite/exercise/next links, and existing anchors. Run `mkdocs build --strict` for site changes. Run targeted executable checks only if code or fixtures change. This plan-only audit does not require a site build because `plans/` is outside the site.
- **Definition of done amendment:** §8’s six items are editorial checks, not proof of improved attention. Permit a concrete role or operational decision instead of a named person, retain correct technical claims and observed-vs-invented distinctions, and require pilot feedback before scaling. Read the rendered openings and closings and then complete an exercise; the serial-only author test cannot assess comprehension or code fidelity.
