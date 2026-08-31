# Reasoning self-checks

End-of-week checks for the required path (0–17) and the optional DL pictures (18–20). These are not trivia. Write your answer first, then open the collapsed block.

Use them with the week's exercise: **predict → run → compare → explain**.

## Week 0 — Strong Python {#week-0-strong-python}

??? question "Predict: `add_tag` uses `tags: list[str] = []`. After two calls with different users, what is in the second list?"

    Both tags. The default list is created once, at function definition, and shared. The second call mutates the same object.

??? question "Diagnose: `MeanBaseline().predict(3)` raises. What did the caller forget, and why is that the right failure?"

    They never called `fit`. A model that invents a mean of 0 is a handler that 200s an empty payload. Raise, don't guess.

??? question "Choose: you need to send a customer row to a JSON API. Dataclass instance, `__dict__`, or a dedicated `to_payload()`?"

    `to_payload()` (or `asdict` with an explicit allowlist). `__dict__` can leak private fields. The API wants JSON types, not a Python object.

??? question "Defend: why is `MeanBaseline` a class with `fit` / `predict` instead of a function `mean(y)`?"

    The same object has to remember what it learned and answer later, possibly in another process after `joblib`. That is sklearn's contract and Week 15's pickle.

## Week 1 — NumPy {#week-1-numpy}

??? question "Predict: CloudWave usage is right-skewed. Will mean feature-total sit above or below the median?"

    Above. Whales pull the mean. That is why the exercise also prints median and p90.

??? question "Diagnose: `mat / row_means` dies with “operands could not be broadcast.” What do you print first?"

    `.shape`. You want `(users, features) / (users, 1)`. A 1-D `(users,)` mean aligns from the right and fights the columns.

??? question "Choose: group usage by `feature_name` in a Python loop, in Pandas, or as a NumPy array from the start?"

    Group in Pandas (ragged keys), then `.to_numpy()` for the slab stats. NumPy is not a dict.

??? question "Defend: why sample the user × feature pivot instead of materializing every user?"

    The lesson is broadcasting, not RAM. A sample still has the shape `(users, features)`. The full join belongs in SQL / Week 3.

## Week 2 — Pandas

??? question "Predict: you `merge(subscriptions, user_events, on='user_id')` with no aggregate. About how many rows come out?"

    Far more than ~49k — events are many-per-user. Output rows explode. Sum(MRR) after that join is a lie (Week 17 incident 1).

??? question "Diagnose: a left-join to “primary region” leaves 8% null region. Bug or segment?"

    Segment until proven otherwise. Users with no events are real. Report the null share; don't silently fill `"NA"` without saying so.

??? question "Choose: grain of the Customer 360 — one row per event, per session, or per user?"

    Per user, for the churn table. Anything else is a different product.

??? question "Defend: why raise if output rows > 1.01 × input rows?"

    A 1% fan-out is already a grain bug. Catching it in five lines is cheaper than debugging a 2× MRR dashboard next quarter.

## Week 3 — as_of {#week-3-as_of}

??? question "Predict: `load_customer_360()` vs `build_features(as_of='2024-06-01')` — which one still sees July usage?"

    `load_customer_360`. It is a convenience sample, not an as-of extract. The pipeline path cuts usage and events at `as_of`.

??? question "Diagnose: `len(build_features(..., n=8000))` ≠ at-risk subscriptions. What broke the grain test?"

    Sampling. Grain tests use `n=None`. Laptop training uses `n=8000`.

??? question "Choose: model input — `tenure_days` or `tenure_so_far`?"

    `tenure_so_far`. Lifetime `tenure_days` already knows who left (or was snapshotted at 2024-11-30).

??? question "Defend: why is an `as_of` after 2024-11-30 illegal in this universe?"

    Usage, events, and billing clip there. You would be asking the warehouse a question it cannot answer — Week 8 calls the cousin of this *censoring*.

## Week 4 — Charts

??? question "Predict: if the y-axis of plan-churn starts at 14% instead of 0, what does a screenshot imply?"

    That free vs paid is a cliff. It is a ~2× ratio on a small base. Truncated axes manufacture urgency.

??? question "Diagnose: region churn bars were built on event rows. Chatty regions look like they churn more. What is wrong?"

    Grain. Collapse to one region per user, then take the mean of `is_churned`.

??? question "Choose: title is “Churn by plan” or “Free churn is ~2× paid”?"

    The claim. A chart is an API response; the title is the JSON field the PM will quote.

??? question "Defend: why annotate the winner instead of using a legend-only color?"

    Color is a hint. The number is the payload. Colorblind CSMs still need the call.

## Week 5 — Signal vs noise {#week-5-signal-vs-noise}

??? question "Predict: drop `free` from the plan × churn table. Does chi-squared still fire at α=0.05?"

    Often it weakens a lot. Most of the original table's drama was free vs paid, not starter vs enterprise. Write the guess *before* you run.

??? question "Diagnose: sentiment t-test p is tiny, histograms overlap heavily. Ship a “bugs make people sad” launch?"

    No. p-values detect differences, not importance. Look at the overlap and the product threshold.

??? question "Choose: 16% vs 20% conversion, equal n. Do you staff an A/B at n=100 or wait for ~400–1000?"

    Wait. The simulation in the exercise is there so you feel how often n=100 “wins” by luck.

??? question "Defend: why is a ranker not a lever?"

    Plan × churn is observational — people chose their plan. A p-value does not license “if we upgrade them they will stay.” That needs an experiment (Week 11's causal trap again).

## Week 6 — Features as API

??? question "Predict: a feature uses behavior from seven days *after* `as_of`. Offline holdout AUC? Production AUC?"

    Offline goes up (time machine). Production falls apart on customers who have not lived those seven days. That is leakage, not a better model.

??? question "Diagnose: scaler fitted on all rows, `mrr` mean differs by $0.12 from train-only. Harmless?"

    The number is tiny on 8k rows. The habit is still wrong. On a time-split or a small segment it will not be tiny.

??? question "Choose: keep `has_usage` *and* `total_usage`, or pick one?"

    Keep both if zero vs missing is a different story (never logged in vs lurker). Otherwise you duplicated a column.

??? question "Defend: `assert_score_payload` instead of another sklearn transformer?"

    Production receives JSON, not a DataFrame. The contract belongs in a function the handler runs. Week 15's `validate()` is this idea with teeth.

## Week 7 — Classification

??? question "Predict: majority dummy accuracy on ~6.4% churn? Dummy AUC?"

    Accuracy ≈ 93–94%. AUC ≈ 0.5. Accuracy is a trap; ranking quality is not.

??? question "Diagnose: threshold 0.8 flags 12 people, precision looks amazing, CS asked for 100 names. What did you optimize?"

    A vanity cutoff. Optimize flagged count (budget) first, then precision at that budget.

??? question "Choose: ship threshold 0.5 or the cut that flags 100 test users?"

    The budget cut. CS has a desk, not a textbook.

??? question "Defend: ablating `tenure_so_far` dropped AUC 0.04. Is it a leak like `tenure_days`?"

    Not the same leak. Tenure-so-far is knowable at noon on `as_of`. It is circular-ish for brand-new signups (they have not had time to churn) but it is not the answer key. Lifetime `tenure_days` is.

## Week 8 — Labels {#week-8-labels}

??? question "Predict: raise threshold 0.5 → 0.8 on a rare-event model. Precision? Recall? Why sharper than 50/50?"

    Precision up, recall down. Rare positives run out fast — you have tens of hits, not thousands — so the swing is violent.

??? question "Diagnose: observation_end = as_of + 10 days, horizon 30. Lots of NaNs, a handful of 1s. Censoring or a bug?"

    Censoring, per row. Cancels you already saw inside those 10 days stay 1. Everyone else you have not watched through the horizon is NaN.

??? question "Choose: Monday email number — ROC-AUC, PR-AUC, or precision@80 on horizon-30?"

    PR-AUC vs dummy, plus precision@budget on a label with enough positives (eventual, in this fixture). Horizon-30 precision@80 is a lottery (~48 positives in the whole file).

??? question "Defend: why must `validate()` reject `churn_date` on the payload?"

    Because that field is the answer key. If it can enter, someone will train on it. Unknown keys are a 400, not a warning.

## Week 9 — Regression

??? question "Predict: `log1p` target, `expm1` predictions. MAE in dollars — better, worse, or only kinder to whales?"

    Often a bit better overall because whales stop dominating the squared/absolute loss. Always report MAE on the original scale.

??? question "Diagnose: overall MAE looks fine, enterprise MAE is 3× free. What did the average hide?"

    A slice. Residuals by plan (and by MRR band) are the product review. Overall MAE is a press release.

??? question "Choose: target is `mrr * tenure_so_far / 30`. Features include both. Ship?"

    Delete it. That R² is a tautology. Lifetime `tenure_days` in the same spot is worse — it knows who left.

??? question "Defend: why is a residual trumpet on high MRR expected?"

    Variance scales with the mean in dollars. A model that is “±$5 on free and ±$5 on enterprise” is lying about one of those groups.

## Week 10 — Clustering

??? question "Predict: unscaled K-Means on `mrr` + `n_support`. What are the clusters actually sorted by?"

    MRR. Dollars dwarf counts. Personas will be “cheap / expensive,” not “needs support.”

??? question "Diagnose: two of four clusters got the same marketing name. Next step?"

    Merge them. K=4 was a teaching default, not a discovered truth.

??? question "Choose: train the churn classifier on cluster id, or on the original columns?"

    Original columns. Cluster id is lossy compression. Churn *rate* by cluster is a story, not a feature.

??? question "Defend: why isn't a persona an API?"

    It has no schema, no stability guarantee, and no `validate()`. You can name a pile. You cannot version a vibe.

## Week 11 — Ranking

??? question "Predict: as k goes 20 → 80 → 200, precision@k goes…?"

    Down (usually). You are digging further into a ranked list. Slack CS accordingly if they 4× the budget.

??? question "Diagnose: you changed k from 80 to 47 after seeing precision. What is that called?"

    p-hacking. k=80 was pre-registered because that is the desk. Leave it.

??? question "Choose: ship the GBT list if it ties `ORDER BY n_support` at precision@80?"

    Don't. Beat a SQL sort or keep the sort. Complexity needs a lift.

??? question "Defend: “usage predicts churn, so force the tutorial.” Reply in two sentences."

    Prediction is not causation. Low usage may *mark* people already leaving; forcing a tutorial can annoy them without moving the label.

## Week 12 — PCA

??? question "Predict: color the PC1–PC2 scatter by `is_churned`. Corner or sprinkle?"

    Usually sprinkle. Churn is not a 2-D blob on this table. If you see a corner, check whether tenure/MRR is just PC1.

??? question "Diagnose: a whale is huge on PC1 but reconstructs well at k for 80% variance. High residual?"

    Not necessarily. A point can sit far along a kept axis and still be in-subspace. Residuals find people the k-D plane *cannot draw*, not the whales.

??? question "Choose: Slack says “PC3 is important (8% variance).” You send…?"

    “PC3 is 8% of *column variation*, not 8% of churn, and we will not staff a project on an unnamed direction.”

??? question "Defend: scale before PCA?"

    Otherwise MRR (dollars) eats PC1 and `n_support` (counts) never gets a vote. Same lesson as unscaled K-Means.

## Week 13 — Ensembles

??? question "Predict: `max_depth=8`, `n_estimators=80` on this table. Train AUC vs test AUC?"

    Train heads toward 1.0; test lags. That gap is the overfit you were asked to see.

??? question "Diagnose: importances look like a shuffle across one-hot plan dummies. Broken forest?"

    Not necessarily. Importances split across dummies. Sum them per original column before you tell a story.

??? question "Choose: teammate wrote “stacking classifier” for `VotingClassifier(voting='soft')`. Correct name?"

    Soft voting — average predicted probabilities. Stacking trains a second model on those predictions.

??? question "Defend: GBT still ships for CloudWave churn this quarter. Why not a net?"

    Mixed types, missingness, ~8k rows, need importances and a pickle. Week 14's net has to beat this, not replace it for fashion.

## Week 14 — Neural nets

??? question "Predict: MLP with `activation='identity'` vs logistic regression AUC?"

    They rhyme. Without ReLU you built a linear model with extra matrix multiplies.

??? question "Diagnose: you commented out `opt.zero_grad()`. Loss explodes. Why?"

    Gradients accumulate across steps. Each `backward()` adds to `.grad` instead of replacing it. That is a running sum, not momentum.

??? question "Choose: hidden sizes (128, 128, 128) on 7 tabular columns. Ship?"

    No. Train AUC inflates, test does not. Capacity without data is a memorizer.

??? question "Defend: five lines to a VP — why the churn model stays a GBT."

    Tabular mixed types, small data, leakage controls we already have, importances CS can argue with, and a net has not beaten the tree on the time split. Revisit if we get sequences or text as *the* signal.

## Week 15 — The pickle {#week-15-the-pickle}

??? question "Predict: shuffled split AUC vs signup_date 80/20 AUC — which is higher?"

    Shuffled, usually. It lets tomorrow's mix into today's fit. The pickle has to live in time.

??? question "Diagnose: `validate({..., 'email': 'ada@cloudwave.test'})` raises. Good or a missing feature?"

    Good. Unknown keys are the PII fence. Email does not belong in `FEATURE_COLS`.

??? question "Choose: threshold 0.5 or the 80th-highest test score?"

    The budget cut. `flag_for_cs` is a staffing bit, not a probability certificate.

??? question "Defend: dump the whole Pipeline, not a naked GradientBoostingClassifier."

    Prod must one-hot `plan_type` the same way train did. Two copies of feature math is Week 17's silent NaN waiting to happen.

## Week 16 — The job {#week-16-the-job}

??? question "Predict: you run `train` twice with the same `--as-of`. Does `artifacts/prod` change?"

    No. Train writes `artifacts/<date>/`. Only `promote` may copy to prod, and only if the gate passes.

??? question "Diagnose: `pr_auc` is below `dummy_pr_auc`. Promote anyway because AUC is 0.71?"

    Refuse. The gate is PR-AUC vs dummy (and vs prod). ROC-AUC on a rare event is how you ship a coin flip with a nice chart.

??? question "Choose: Airflow this week, or a five-line cron?"

    Cron. Airflow when the cron file is boring — the lesson's ship rule.

??? question "Defend: who may write `artifacts/prod`?"

    `promote.py`, after `gate()`. Not train, not a notebook, not a human `cp` on Friday.

## Week 17 — On-call

??? question "Predict: incident 1 (join doubled MRR). First log line — AUC or row counts?"

    Row counts. Unique `user_id` vs training rows. AUC still looks great when you duplicated whales.

??? question "Diagnose: half of tonight's scores are exactly 0.5. Which incident class?"

    Silent NaN / fill-value divergence. `validate()` should have rejected the payload. A quiet 0.5 is a skipped contract.

??? question "Choose: the bot wants to know if `user_041906` will cancel. Prose, or `get_churn_score`?"

    The function. The bot is a client of the pickle, not a second churn model.

??? question "Defend: `allowed_tools` returns `[]` for “skip the allowlist.” Why not a system prompt?"

    Prompts are suggestions. The allowlist is code. Golden tickets make widening it a CI failure.

## Weeks 18-20 — DL pictures (optional) {#weeks-18-20-dl-pictures-optional}

These weeks build intuition for modern systems. They do **not** teach research-level training from scratch.

??? question "Predict (18): a `Linear(12, 1)` ties your 1-D CNN on weekly usage. What was the signal?"

    The *total*, not the *shape*. The stencil had nothing to find. Stop claiming the CNN saw a late-week drop.

??? question "Diagnose (18): why is a CNN the wrong tool for a Customer 360 row (`mrr`, `plan_type`, …)?"

    There is no axis to slide over. Convolution wants translation invariance on a grid or a sequence. Use a tree.

??? question "Choose (19): last hidden state vs `out.mean(dim=1)` if you care about a dip in week 4 of 12."

    Mean (or an attention pool). The last step can forget week 4 unless the clipboard carried it on purpose.

??? question "Defend (19): `torch.flip` on the weeks kills accuracy. What did you just prove?"

    The model used *order*, not just the sum. If flip had been a no-op, you built a fancy total-usage detector.

??? question "Predict (20): comment out positional embeddings. What happens to accuracy, and why?"

    It should drop (or ignore order). Without positions, attention is a bag of tokens — `login failed` and `failed login` look the same.

??? question "Diagnose (20): someone indexes `self.enc.layers[0].self_attn` on raw character ids. What is wrong?"

    Those ids were not embedded as the 3-token toy (`login` / `failed` / `again`). Print the toy weights, or embed first.

??? question "Choose (20): CloudWave churn on seven table columns — CNN, RNN, Transformer, or GBT?"

    GBT. Sequences and text are the other three. After week 20 you should be able to *read* a transformer block, not train GPT from scratch.

??? question "Defend (18–20): you will not implement FlashAttention or debug CUDA kernels. What should you be able to do?"

    Explain convolution, recurrence, and attention as software analogies; know when a tree still wins on SaaS tables; read a high-level transformer block diagram. Next step if you want the research path: Karpathy, fast.ai, a standard DL course — not this syllabus.

## How to use this page

One check per sitting is enough. If you cannot defend the answer out loud, reread the week's **Think of it like…** box, not the formula.
