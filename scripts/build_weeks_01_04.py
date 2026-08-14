#!/usr/bin/env python3
"""Rebuild Weeks 1–4 for engineers without a math background."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbformat_util import BOOT, LAPTOP_BOX, code_cell, md_cell, write_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def week1():
    cells = [
        md_cell(
            """# Week 1 — NumPy: Fast Math on Whole Columns

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Software engineers. You do not need calculus or linear algebra.

You already have Python (Week 0). This week is not “become a numerical analyst.” It is: **stop writing Python `for` loops over 160,000 usage events.**

---

## 🎯 What you will be able to do

- Treat a NumPy array like a typed column in a database — one type, one block of memory
- Replace a loop with a single vectorized operation
- Picture **broadcasting** as “stamp this row down the table”
- Read a **z-score** as “how weird is this user, in units of typical spread”
- Know when to stay in Pandas / SQL instead

## How to read this course

Every week uses the same colored boxes:

| Box | Meaning |
|---|---|
| **Think of it like…** | Everyday or software analogy. Start here. |
| **Engineer mental model** | How this shows up in a codebase. |
| **Watch out** | The foot-gun of the week. |
| **Ship / don’t ship** | Decision rule, not theory. |
| **Math, translated** | One formula, immediately in English. |
"""
        ),
        code_cell(BOOT),
        md_cell(
            """## 🏢 Scenario — CloudWave’s daily health numbers

You are on the growth team at **CloudWave**. The CEO wants a daily pulse:

- Which features are actually used?
- Who are the top 10% power users?
- Which usage numbers look “weird” enough to investigate?

The raw file `feature_usage.csv` has **160,000 rows**. A Python loop will work. It will also be the slowest part of your job.

<div class="think-box">
<strong>Think of it like… SIMD, SQL, or Excel — not “math class.”</strong>
<p>A NumPy array is a <em>single typed buffer</em>, like <code>int64_t usage[160000]</code> in C, or one column in Postgres. When you write <code>usage * 2</code>, you are not looping in Python. You are asking the CPU to map an operation over the whole column — the same idea as <code>SELECT usage * 2 FROM events</code>.</p>
</div>
"""
        ),
        md_cell(
            """## Visual: Python list vs NumPy array

A Python list of numbers is a list of **pointers to objects**. A NumPy array is a **contiguous block** of raw values.

```
Python list  [ 1,  2,  3,  4 ]
               │   │   │   │
               ▼   ▼   ▼   ▼
            int  int  int  int     ← 5 objects, extra headers, cache misses

NumPy array   [ 1 | 2 | 3 | 4 ]   ← one slab, one type, CPU-friendly
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p><code>list</code> = JSON array of values (flexible, slow). <code>ndarray</code> = a column in a parquet file (rigid, fast). If every cell is the same type and you want the same operation on every row, use the slab.</p>
</div>
"""
        ),
        code_cell(
            """# Proof, not a slogan: loop vs vectorized multiply on 1 million numbers
import time

n = 1_000_000
py_list = list(range(n))
np_arr = np.arange(n)

t0 = time.perf_counter()
_ = [x * 2 for x in py_list]
loop_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
_ = np_arr * 2
vec_ms = (time.perf_counter() - t0) * 1000

fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.barh(["Python list comprehension", "NumPy vectorized"], [loop_ms, vec_ms],
               color=["#f59e0b", "#3b82f6"])
ax.set_xlabel("Milliseconds to double 1,000,000 numbers")
ax.set_title("Same work. Different runtime.")
for bar, val in zip(bars, [loop_ms, vec_ms]):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f} ms", va="center")
plt.tight_layout()
plt.show()

print(f"Speedup: {loop_ms / max(vec_ms, 1e-9):.0f}×")
print("You do not need to remember the number. Remember the shape of the bar chart.")
"""
        ),
        md_cell(
            """## Worked example — real CloudWave usage

We will load `usage_count` as a NumPy array and ask ordinary product questions.

<div class="cue-box">
<strong>Visual cue — “one column, many questions”</strong>
<p>Once the numbers live in an array, mean / median / percentile are the same <em>kind</em> of operation: walk the column once in compiled code.</p>
</div>
"""
        ),
        code_cell(
            """usage = pd.read_csv(DATA / "feature_usage.csv")
print(usage.head(3))
print(f"\\n{len(usage):,} rows  |  columns: {list(usage.columns)}")

counts = usage["usage_count"].to_numpy()  # the slab
print(f"array dtype={counts.dtype}  shape={counts.shape}")

print("\\nHow 'busy' is a typical event row?")
print(f"  mean   {counts.mean():.2f}   ← pulled up by heavy users")
print(f"  median {np.median(counts):.2f}   ← a better 'typical'")
print(f"  p90    {np.percentile(counts, 90):.0f}")
print(f"  p99    {np.percentile(counts, 99):.0f}")
print(f"  max    {counts.max():.0f}")

fig, ax = plt.subplots(figsize=(8, 3.4))
# Cap the x-axis so a few whales do not flatten the picture
cap = np.percentile(counts, 99)
ax.hist(counts[counts <= cap], bins=40, color="#6366f1", edgecolor="white")
ax.axvline(np.median(counts), color="#b45309", lw=2, label=f"median {np.median(counts):.0f}")
ax.axvline(np.percentile(counts, 90), color="#047857", lw=2, ls="--",
           label=f"p90 {np.percentile(counts, 90):.0f}")
ax.set_title("Usage counts are skewed — the mean is not 'typical'")
ax.set_xlabel("usage_count (capped at p99 for readability)")
ax.set_ylabel("rows")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## Broadcasting — stamping a row down a table

Broadcasting sounds like linear algebra. It is closer to **CSS stretching** or **SQL joining a one-row lookup onto every day**.

Imagine a week of daily active users for 4 regions, and a “normal Tuesday” baseline per region:

```
DAU (7 days × 4 regions)          baseline (4 regions)
Mon  [ 80  90  70  60 ]           [ 100  100  80  70 ]
Tue  [110  95  85  72 ]      ÷
...
```

NumPy **stamps** the 4-number baseline onto every day, then divides. You never write a nested loop.

<div class="math-box">
<strong>Math, translated</strong>
<p>Shapes line up <em>from the right</em>. A <code>(7, 4)</code> table and a <code>(4,)</code> row are compatible because the last dimension matches. The length-4 row is repeated 7 times. That repetition is broadcasting. It does not copy the array 7 times in your head — the library does it.</p>
</div>
"""
        ),
        code_cell(
            """# Tiny picture you can see — 7 days × 4 regions
rng = np.random.default_rng(7)
regions = ["NA", "EMEA", "APAC", "LATAM"]
dau = rng.integers(60, 140, size=(7, 4))
baseline = np.array([100, 100, 80, 70])  # "normal" DAU per region

ratio = dau / baseline  # (7, 4) / (4,) → stamped automatically

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, grid, title, cmap in [
    (axes[0], dau, "Raw DAU", "Blues"),
    (axes[1], np.tile(baseline, (7, 1)), "Baseline stamped 7×", "Oranges"),
    (axes[2], ratio, "DAU / baseline", "RdYlGn"),
]:
    im = ax.imshow(grid, cmap=cmap)
    ax.set_xticks(range(4), regions, fontsize=8)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}" if grid is ratio else f"{int(grid[i, j])}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

print("Hottest cell = that region+day beat its own baseline the most.")
print(f"Best day/region: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][np.unravel_index(ratio.argmax(), ratio.shape)[0]]} / {regions[np.unravel_index(ratio.argmax(), ratio.shape)[1]]}")
"""
        ),
        md_cell(
            """## Z-score — “how weird is this user?”

Product people say “this account is an outlier.” Engineers should say **how many typical-spreads away from the middle**.

<div class="math-box">
<strong>Math, translated</strong>
<p><code>z = (value − mean) / std</code> → “this user is z typical-spreads from average.” <code>|z| &gt; 2.5</code> is a useful hunt-list, not a law of nature. It is a smoke alarm, not a guilty verdict.</p>
</div>

<div class="watch-box">
<strong>Watch out</strong>
<p>On a skewed usage column the mean is dragged by whales, so z-scores flag even more whales. For “unusual,” start with <strong>percentiles</strong> (top 1%). Use z-scores after a log, or on something already bell-shaped.</p>
</div>
"""
        ),
        code_cell(
            """per_user = usage.groupby("user_id")["usage_count"].sum().to_numpy()
z = (per_user - per_user.mean()) / per_user.std()

print(f"Users: {len(per_user):,}")
print(f"Top 1% usage threshold: {np.percentile(per_user, 99):,.0f}")
print(f"|z| > 2.5 users: {(np.abs(z) > 2.5).sum():,}")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(z, bins=40, color="#8b5cf6", edgecolor="white")
ax.axvline(2.5, color="#b91c1c", ls="--", label="|z| = 2.5")
ax.axvline(-2.5, color="#b91c1c", ls="--")
ax.set_title("Z-score of total usage per user — right tail is the whale list")
ax.set_xlabel("z-score")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## When NumPy is the wrong tool

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<ul>
<li><strong>Use NumPy</strong> for one typed column (or a grid) and the same math on every cell: scores, percentiles, normalizations.</li>
<li><strong>Use Pandas</strong> when rows have names, joins, group-bys, missing values, dates — that is next week, and it is most of SaaS work.</li>
<li><strong>Use SQL / Spark / DuckDB</strong> when the table is bigger than RAM.</li>
<li><strong>Stay in Python</strong> for strings, nested JSON, one-off business rules.</li>
</ul>
</div>

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Pandas is your ORM + dataframe. NumPy is the compute kernel hiding under <code>.values</code> / <code>.to_numpy()</code>. You will bounce between them all course. That is normal.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises (use the real files — no fake 90×50 DAU grid)

**1. Feature ranking.** Load `feature_usage.csv`. For each `feature_name`, compute total `usage_count` with a group-by, then convert the totals to a NumPy array and print mean / median / p90 of *those feature totals*.

**2. Broadcasting on a real pivot.** Pivot a *sample* of users × features into a 2-D usage matrix (`fillna(0)`). Divide each row by that row’s mean (user-normalized usage). Shapes: `(users, features) / (users, 1)`.

**3. Whale hunt.** Per `user_id`, sum usage. List user ids in the top 1%. How many are they? What share of all usage do they account for?

<details class="lesson-extra">
<summary>💡 Hint — row-normalize with broadcasting</summary>

```python
mat = pivot.to_numpy()
row_means = mat.mean(axis=1, keepdims=True)  # shape (n_users, 1)
normalized = mat / np.where(row_means == 0, 1, row_means)
```

`keepdims=True` is what makes the stamp work. A bare `(n_users,)` would not line up from the right.
</details>
"""
        ),
        md_cell(
            """## 🤔 Reflection

1. A teammate writes a `for user in users` loop to compute mean usage. When do you leave it, and when do you rewrite it?
2. Why can the **mean** of CloudWave usage lie to a PM, while the **median** and **p90** tell a better story?
3. Broadcasting failed with `operands could not be broadcast`. What do you inspect first? (Print `.shape`. Align from the right.)

## 🔗 Next week

Pandas: joins, group-bys, and building a Customer 360 table — the bread and butter of SaaS analytics. NumPy stays underneath.
"""
        ),
    ]
    write_notebook(OUT / "week-01-saas.ipynb", cells, "Week 1 — NumPy")


def week2():
    cells = [
        md_cell(
            """# Week 2 — Pandas: SQL You Already Know, in Python

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written `SELECT / JOIN / GROUP BY`. You do not need statistics yet.

---

## 🎯 What you will be able to do

- Map every Pandas verb to the SQL you already know
- Build a **Customer 360** row from four messy systems
- Catch a join that secretly exploded 10×
- Decide what to do with missing values (0 vs median vs “missing is a signal”)

<div class="think-box">
<strong>Think of it like… a typed DataFrame is a SQL table that lives in RAM.</strong>
<p>A Series is a column. An index is a primary key (sometimes sloppy). <code>merge</code> is <code>JOIN</code>. <code>groupby</code> is <code>GROUP BY</code>. If you can write the query, you can write the Pandas.</p>
</div>
"""
        ),
        code_cell(BOOT),
        md_cell(
            """## 🏢 Scenario — four systems, one customer

CloudWave’s data is not one warehouse table. It is:

| System | File | Grain |
|---|---|---|
| Billing | `subscriptions.csv` | one row per user |
| Product | `feature_usage.csv` | one row per user × feature × day |
| Telemetry | `user_events.csv` | one row per event |
| Support | `feedback.json` | one JSON object per comment (JSON Lines) |

Your job: **one row per customer** the ML weeks can train on.

```
subscriptions ──┐
feature_usage ──┼──► customer_360  (one row = one user)
user_events  ───┤
feedback     ───┘
```
"""
        ),
        md_cell(
            """## SQL → Pandas cheat sheet

| You already write | Pandas |
|---|---|
| `SELECT cols` | `df[["user_id", "mrr"]]` |
| `WHERE mrr > 50` | `df[df["mrr"] > 50]` |
| `ORDER BY mrr DESC` | `df.sort_values("mrr", ascending=False)` |
| `GROUP BY plan_type` | `df.groupby("plan_type")` |
| `COUNT(*)` | `.size()` or `.agg(n=("user_id", "count"))` |
| `LEFT JOIN` | `left.merge(right, on="user_id", how="left")` |
| `COALESCE(x, 0)` | `df["x"].fillna(0)` |

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Prefer <strong>named aggregations</strong> over chained mystery columns. Treat <code>merge</code> like a code review: assert row counts before and after, the same way you would check a SQL join in a PR.</p>
</div>
"""
        ),
        md_cell(
            """## Visual: what each join keeps

```
LEFT (subscriptions)     RIGHT (feedback)

 user_1 ●───────────● comment_a     INNER: only matches
 user_2 ●                           LEFT:  keep user_2, feedback = NaN
 user_3 ●───────────● comment_b     OUTER: everyone, holes filled with NaN
                    ● comment_orphan
```

**SaaS default is LEFT JOIN from the customer table.** You do not want to drop a paying user because they never left feedback.
"""
        ),
        code_cell(
            """subs = pd.read_csv(
    DATA / "subscriptions.csv",
    usecols=["user_id", "plan_type", "mrr", "signup_date", "churn_date", "is_churned", "tenure_days"],
    parse_dates=["signup_date", "churn_date"],
)
usage = pd.read_csv(
    DATA / "feature_usage.csv",
    usecols=["user_id", "feature_name", "usage_count", "avg_session_seconds", "date"],
    parse_dates=["date"],
)
events = pd.read_csv(
    DATA / "user_events.csv",
    usecols=["event_id", "user_id", "event_type", "timestamp", "device", "region"],
    parse_dates=["timestamp"],
)
# feedback.json is JSON Lines (one object per line), not a JSON array
feedback = pd.read_json(DATA / "feedback.json", lines=True)

print("subscriptions", subs.shape, list(subs.columns))
print("feature_usage", usage.shape, list(usage.columns))
print("user_events   ", events.shape, list(events.columns))
print("feedback      ", feedback.shape, list(feedback.columns))
print("\\nplan_type counts:\\n", subs["plan_type"].value_counts().to_string())
print("churn rate   ", subs["is_churned"].mean().round(3))
"""
        ),
        md_cell(
            """## Worked example — Customer 360

Aggregate the *many* side down to *one row per user* **before** you join. That is the single most important ETL habit in this course.
"""
        ),
        code_cell(
            """# 1) collapse usage and events to user grain
usage_by_user = usage.groupby("user_id").agg(
    total_usage=("usage_count", "sum"),
    features_adopted=("feature_name", "nunique"),
    last_usage_date=("date", "max"),
).reset_index()

events_by_user = events.groupby("user_id").agg(
    total_events=("event_id", "count"),
    n_devices=("device", "nunique"),
    n_regions=("region", "nunique"),
).reset_index()

feedback_by_user = feedback.groupby("user_id").agg(
    n_feedback=("feedback_text", "count"),
    avg_sentiment=("sentiment_score", "mean"),
).reset_index()

print("before join, subscriptions rows:", len(subs))

customer = (
    subs.merge(usage_by_user, on="user_id", how="left")
        .merge(events_by_user, on="user_id", how="left")
        .merge(feedback_by_user, on="user_id", how="left")
)

print("after  join, customer rows:     ", len(customer))
print("row-count ratio (want ~1.0):    ", round(len(customer) / len(subs), 3))

# missingness is information
customer["has_feedback"] = customer["n_feedback"].fillna(0).gt(0).astype(int)
customer["total_usage"] = customer["total_usage"].fillna(0)
customer["total_events"] = customer["total_events"].fillna(0)
customer["features_adopted"] = customer["features_adopted"].fillna(0)

print("\\nCustomer 360 sample:")
print(customer[["user_id", "plan_type", "mrr", "is_churned",
                "total_usage", "features_adopted", "total_events",
                "has_feedback"]].head())
"""
        ),
        md_cell(
            """## Watch the join explode on purpose

If you join subscriptions to **raw** feature_usage (many rows per user), you duplicate every billing field.

<div class="watch-box">
<strong>Watch out — fan-out</strong>
<p>A 50,000-row customer table joined to 160,000 usage rows becomes ~160,000 rows, and <code>mrr.sum()</code> will lie by a factor of ~3. Always aggregate the many-side first. Always print <code>len(left)</code> vs <code>len(result)</code>.</p>
</div>
"""
        ),
        code_cell(
            """exploded = subs.merge(usage[["user_id", "usage_count"]], on="user_id", how="left")
print(f"subscriptions: {len(subs):,}")
print(f"joined to raw usage: {len(exploded):,}   ← {len(exploded)/len(subs):.1f}× blow-up")
print(f"true total MRR:     ${subs['mrr'].sum():,.0f}")
print(f"exploded MRR sum:   ${exploded['mrr'].sum():,.0f}   ← do not ship this number")
"""
        ),
        md_cell(
            """## Missing values — a decision, not a default

```
Is “missing” actually zero?
   │
   ├─ YES (they never used the feature) → fillna(0)
   │
   └─ NO (we never observed it)
         │
         ├─ The hole itself predicts the outcome → keep a has_* flag
         └─ The model needs a number            → median of the *training* set
```

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p><code>fillna(0)</code> on <em>usage</em> is honest: no events means no usage. <code>fillna(0)</code> on <em>sentiment</em> is a lie: “no review” is not “neutral review.” Use a flag.</p>
</div>
"""
        ),
        code_cell(
            """print("Null share in customer_360 (after left joins, before our fills):")
probe = (
    subs.merge(usage_by_user, on="user_id", how="left")
        .merge(feedback_by_user, on="user_id", how="left")
)
print((probe.isna().mean() * 100).round(1).astype(str) + "%")

print("\\nChurn rate by 'left any feedback':")
probe["has_feedback"] = probe["n_feedback"].notna()
print(probe.groupby("has_feedback")["is_churned"].mean().round(3))
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Plan snapshot.** Churn rate, mean MRR, and user count by `plan_type`. Which plan is the leaky bucket?

**2. Region mix.** From `user_events`, each user’s most-common `region`. Left-join onto subscriptions. Does churn differ by region?

**3. Quality check.** Write a 5-line join validator: input rows, output rows, duplicate `user_id` count, null share of a key metric, and a `raise` if output rows > 1.01 × input rows.

<details class="lesson-extra">
<summary>✅ One possible plan snapshot</summary>

```python
subs.groupby("plan_type").agg(
    users=("user_id", "count"),
    churn_rate=("is_churned", "mean"),
    arpu=("mrr", "mean"),
).sort_values("churn_rate", ascending=False)
```
</details>
"""
        ),
        md_cell(
            """## 🤔 Reflection

1. Your exploded MRR was 3× too big. What code review comment do you leave?
2. A PM says “customers who write feedback churn less.” Is that product magic, or selection (happy people write reviews)?
3. When would you *want* an inner join from subscriptions to events?

## 🔗 Next week

Pictures. We will actually plot the dashboard the CFO asked for — not hide it inside a collapsed solution.
"""
        ),
    ]
    write_notebook(OUT / "week-02-saas.ipynb", cells, "Week 2 — Pandas")


def week3():
    cells = [
        md_cell(
            """# Week 3 — Charts That Change a Decision

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who will paste a chart into Slack or a board deck. Not a design class.

---

## 🎯 What you will be able to do

- Pick a chart the way you pick a data structure — by the *question*
- Build a one-page CloudWave dashboard that actually renders
- Read a cohort heatmap without being fooled by “new customers look loyal”
- Spot a dishonest chart (truncated axis, rainbow pie)

<div class="think-box">
<strong>Think of it like… picking an API response shape.</strong>
<p>A line chart is a time series. A bar chart is a group-by. A scatter is a join between two metrics. A heatmap is a group-by on <em>two</em> keys. A pie chart is an unreadable JSON blob with a color theme.</p>
</div>
"""
        ),
        code_cell(BOOT + "\nimport seaborn as sns\nsns.set_theme(style='whitegrid')"),
        md_cell(
            """## 🏢 Scenario — one page for the CFO

Your CFO wants, on one screen:

1. Is churn getting worse?
2. Which plan is the leak?
3. Do engaged customers stay?
4. Do older signup cohorts retain?

<div class="cue-box">
<strong>Visual cue — chart chooser</strong>
<p><strong>Trend over time</strong> → line. <strong>Compare categories</strong> → bar. <strong>Relationship</strong> → scatter. <strong>Two categorical axes + a rate</strong> → heatmap. <strong>Distribution</strong> → histogram or box. <strong>Share of 100%</strong> → still a bar, not a pie (people cannot compare slice angles).</p>
</div>
"""
        ),
        md_cell(
            """## Why not a pie chart

```
Pie:     [/////####......]   which slice is bigger — #### or .....?
Bar:     Pro         ████
         Starter     ██████
         Free        ████████████
```

If the point is “free is the biggest bucket,” a sorted bar wins in 200ms of eye time.
"""
        ),
        code_cell(
            """subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["signup_date", "churn_date"])
usage = pd.read_csv(DATA / "feature_usage.csv")

subs["signup_month"] = subs["signup_date"].dt.to_period("M").dt.to_timestamp()

usage_by_user = usage.groupby("user_id").agg(
    total_usage=("usage_count", "sum"),
    features_adopted=("feature_name", "nunique"),
).reset_index()
df = subs.merge(usage_by_user, on="user_id", how="left")
df["total_usage"] = df["total_usage"].fillna(0)
df["features_adopted"] = df["features_adopted"].fillna(0)

# 1) churn by plan
churn_plan = df.groupby("plan_type")["is_churned"].mean().reindex(
    ["free", "starter", "pro", "enterprise"]
)

# 2) churn by signup month (careful: recent months have had less time to churn)
churn_month = df.groupby("signup_month")["is_churned"].mean()

# 3) engagement vs outcome
sample = df.sample(min(4000, len(df)), random_state=7)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

churn_plan.plot(kind="bar", ax=axes[0, 0], color="#6366f1", rot=0)
axes[0, 0].set_title("Churn rate by plan — free is the leaky bucket")
axes[0, 0].set_ylabel("churn rate")
axes[0, 0].set_ylim(0, 0.15)

churn_month.plot(ax=axes[0, 1], marker="o", color="#0f766e")
axes[0, 1].set_title("Churn by signup month — recent months look 'better'")
axes[0, 1].set_ylabel("churn rate")
axes[0, 1].annotate("haven't had time\\nto churn yet →",
                    xy=(0.72, 0.2), xycoords="axes fraction", color="#b45309")

axes[1, 0].scatter(sample["features_adopted"], sample["is_churned"]
                   + np.random.default_rng(0).normal(0, 0.03, len(sample)),
                   alpha=0.15, s=12, c="#1d4ed8")
axes[1, 0].set_title("Features adopted vs churned (jittered 0/1)")
axes[1, 0].set_xlabel("distinct features used")
axes[1, 0].set_ylabel("churned (jittered)")

# 4) tenure distribution by outcome
df.boxplot(column="tenure_days", by="is_churned", ax=axes[1, 1], grid=False)
axes[1, 1].set_title("Tenure by churn flag — churners leave earlier")
axes[1, 1].set_xlabel("is_churned")
axes[1, 1].get_figure().suptitle("")

plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## Cohort heatmap — the chart that fools smart people

**30-day retention** for the cohort that signed up *last week* is not comparable to the cohort from a year ago. Young cohorts have not had time to die.

<div class="watch-box">
<strong>Watch out</strong>
<p>A heatmap that is bright-green in the bottom rows (newest signups) often means <em>they are still new</em>, not that you suddenly built a better product. Gray-out cells that have not reached that age.</p>
</div>

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>A cohort chart is a two-key group-by: <code>(signup_month, age_bucket) → retained_rate</code>. Same as a SQL cube. The visual is optional; the grain is not.</p>
</div>
"""
        ),
        code_cell(
            """# Simple age-at-observation proxy using tenure_days + churn
# Active customers: tenure is how long they have lived so far
# Churned customers: tenure is how long they lived before leaving

def retention_at(days: int, frame: pd.DataFrame) -> pd.Series:
    # Eligible = old enough to have reached `days`.
    # Still-active users with short tenure have not had time to churn — drop them.
    eligible = frame.copy()
    too_young = (eligible["is_churned"] == 0) & (eligible["tenure_days"] < days)
    eligible = eligible[~too_young]
    retained = eligible["tenure_days"] >= days
    return eligible.assign(retained=retained).groupby("signup_month")["retained"].mean()

months = sorted(df["signup_month"].dropna().unique())
ages = [30, 60, 90]
heat = pd.DataFrame({f"{d}d": retention_at(d, df) for d in ages})

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(heat.tail(18), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0.7, vmax=1)
ax.set_title("Retention by signup month × age — read across a row")
ax.set_ylabel("signup month")
plt.tight_layout()
plt.show()

print("Read a row: of people who signed up that month and are old enough,")
print("what fraction lived at least 30 / 60 / 90 days.")
"""
        ),
        md_cell(
            """## Bad chart vs honest chart

Same data. One lies with a truncated axis.
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
churn_plan.plot(kind="bar", ax=axes[0], color="#ef4444", rot=0)
axes[0].set_ylim(0.03, 0.10)
axes[0].set_title("❌ Dishonest: axis starts at 3%")
axes[0].set_ylabel("churn rate")

churn_plan.plot(kind="bar", ax=axes[1], color="#22c55e", rot=0)
axes[1].set_ylim(0, 0.15)
axes[1].set_title("✅ Honest: axis starts at 0")
axes[1].set_ylabel("churn rate")
plt.tight_layout()
plt.show()

print("The left chart makes starter vs pro look like a crisis.")
print("The right chart says: all paid plans are similar; free is different.")
"""
        ),
        md_cell(
            """<div class="ship-box">
<strong>Ship / don’t ship</strong>
<ul>
<li><strong>Slack / debugging:</strong> default Pandas/Seaborn is fine. Label axes. Start bars at 0.</li>
<li><strong>Board deck:</strong> one sentence title that is the insight, not the chart type. “Free churn is 2× paid,” not “Churn by plan.”</li>
<li><strong>Never:</strong> dual axes, 3-D bars, pie-of-pies, rainbow on unordered categories.</li>
</ul>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Adoption curve.** For users with a `signup_date`, plot average `features_adopted` by `plan_type` as a bar. Annotate the winner.

**2. Region bars.** Most-common region per user from events, then churn rate by region. Horizontal bars, sorted.

**3. Honest title.** Rebuild the plan-churn bar so the title is a claim (“Free churn is ~2× paid”) and the y-axis starts at 0.

## 🤔 Reflection

1. A region’s churn *looks* high. List five non-product reasons (pricing, support hours, competitor, sales quality, data bug).
2. Feature adoption correlates with lower churn. Draw the causation arrow both ways.
3. Which one chart would you send the CEO, and which sentence sits above it?

## 🔗 Next week

The CFO asks: “16% vs 20% churn — is that real?” We will answer without turning you into a statistician.
"""
        ),
    ]
    write_notebook(OUT / "week-03-saas.ipynb", cells, "Week 3 — Visualization")


def week4():
    cells = [
        md_cell(
            """# Week 4 — “Is This Real, or Just Noise?”

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

<div class="think-box">
<strong>Think of it like… a code review, or a courtroom.</strong>
<p>The <strong>null hypothesis</strong> is the boring default: “these two plans churn the same; the difference is luck.” You do <em>not</em> prove the new plan works. You ask: <em>if they were the same, how often would luck produce a gap this big?</em> That frequency is the p-value. Innocent until proven guilty. High bar to convict.</p>
</div>
"""
        ),
        code_cell(BOOT + "\nfrom scipy import stats"),
        md_cell(
            """## 🏢 Scenario — should we roll out Premium?

Early data:

| Plan | Churned | Customers | Rate |
|---|---|---|---|
| Premium | 8 | 50 | **16%** |
| Standard | 12 | 60 | **20%** |

A PM sees “Premium is better.” An engineer asks: **with this few customers, how often would a 4-point gap appear by coin-flip?**

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>A p-value is <em>not</em> “the probability we are wrong.” It is not “the probability Premium is worse.” It is: <strong>how often a world with no real difference produces a result this spicy.</strong> Same idea as “how often would this flaky test fail on a green build?”</p>
</div>
"""
        ),
        md_cell(
            """## Visual: luck can look like a win

We will fake 10,000 worlds where both plans truly churn at 18%. In each world, draw 50 + 60 customers. Plot the Premium − Standard gap. Then mark the gap we actually saw (−4 points).
"""
        ),
        code_cell(
            """rng = np.random.default_rng(42)
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
"""
        ),
        md_cell(
            """## The same answer, with a library test

Chi-squared (or Fisher’s exact, for tiny counts) is the grown-up version of the histogram above.

<div class="math-box">
<strong>Math, translated</strong>
<p>p ≈ 0.03 means: <em>in a no-difference world, about 3 in 100 reruns look this extreme.</em> It does <strong>not</strong> mean “there is a 3% chance Premium is a bad idea.”</p>
</div>
"""
        ),
        code_cell(
            """table = np.array([[8, 42],   # premium: churned, retained
                  [12, 48]])  # standard
chi2, p, dof, expected = stats.chi2_contingency(table)
print("Chi-squared p-value on the 8/50 vs 12/60 story:", round(p, 3))
print("Expected counts if plans were equal:\\n", expected.round(1))
print("\\nVerdict: p is large. We do NOT have enough evidence to declare Premium better.")
print("Ship decision: keep collecting data. Do not rewrite billing based on 110 customers.")
"""
        ),
        md_cell(
            """## Now the full CloudWave table

Same question, real `subscriptions.csv`. More customers → the same 4-point gap would be a much bigger deal.

<div class="cue-box">
<strong>Visual cue — which test?</strong>
<p><strong>Category vs category</strong> (plan × churned) → chi-squared.<br>
<strong>Number vs 2 groups</strong> (MRR for churned vs not) → t-test (or Mann-Whitney if the histogram is a whale-tail).<br>
<strong>Number vs 3+ groups</strong> (usage by region) → ANOVA, then look at the picture before you trust the p.</p>
</div>
"""
        ),
        code_cell(
            """subs = pd.read_csv(DATA / "subscriptions.csv")

ct = pd.crosstab(subs["plan_type"], subs["is_churned"])
print("Counts:\\n", ct)
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\\nChi-squared p-value across all plans: {p:.2e}")

rates = subs.groupby("plan_type")["is_churned"].agg(["mean", "count"])
# Wilson-style interval via the binomial (good enough picture)
cis = []
for plan, row in rates.iterrows():
    lo, hi = stats.binom.interval(0.95, int(row["count"]), row["mean"])
    cis.append((plan, row["mean"], lo / row["count"], hi / row["count"], row["count"]))
ci_df = pd.DataFrame(cis, columns=["plan", "rate", "lo", "hi", "n"]).set_index("plan")
print("\\n95% range of plausible churn rates:")
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
"""
        ),
        md_cell(
            """## A number vs two groups — do churners pay less?

T-test asks: “is the difference in average MRR bigger than the usual jitter in averages?”
"""
        ),
        code_cell(
            """churned = subs.loc[subs["is_churned"] == 1, "mrr"]
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
"""
        ),
        md_cell(
            """<div class="watch-box">
<strong>Watch out</strong>
<ul>
<li><strong>p-hacking:</strong> 20 slices of the data will produce one “p &lt; 0.05” by accident. Pre-register the question, or treat extra slices as exploration.</li>
<li><strong>Significance ≠ importance:</strong> with 50,000 rows, a 0.2% churn gap can be “significant” and still not worth an engineering quarter.</li>
<li><strong>CI overlap</strong> is a sloppy shortcut. Look at the interval on the <em>difference</em>, or just look at dollars.</li>
</ul>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Ship when (1) the interval on the lift is mostly above your <em>business</em> threshold, (2) you have looked at the chart, (3) a second slice (another month, another region) rhymes. p &lt; 0.05 is a filter, not a launch button.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Paid-only chi-squared.** Drop `plan_type == "free"`. Is churn still different across starter / pro / enterprise? Predict the answer before you run it.

**2. Sentiment.** Load `feedback.json` (`lines=True`). Is mean `sentiment_score` different for `category == "bug"` vs `"praise"`? Which test? (Two groups, a number → t-test. Then look at the histogram.)

**3. Sample size gut check.** Keep the 16% vs 20% rates. How many customers per plan (equal n) until a simulation p-value usually drops under 0.05? Try n = 100, 400, 1000.

## 🤔 Reflection

1. Explain a p-value to a PM in one sentence without the word “significant.”
2. Why did 8/50 vs 12/60 fail, while the full table’s plan comparison did not?
3. You ran 12 ad-hoc tests on one Friday. How many “wins” do you expect by luck at α = 0.05?

## 🔗 Next week

Feature engineering — turning Customer 360 columns into the **API contract** of a model, without leaking the future into the past.
"""
        ),
    ]
    write_notebook(OUT / "week-04-saas.ipynb", cells, "Week 4 — Statistics")


if __name__ == "__main__":
    week1()
    week2()
    week3()
    week4()
