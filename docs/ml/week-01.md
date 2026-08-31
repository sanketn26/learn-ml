# Week 1 — NumPy: Fast Math on Whole Columns

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

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.course_data import find_data_dir

DATA = find_data_dir()
```


## 🏢 Scenario — CloudWave’s daily health numbers

You are on the growth team at **CloudWave**. The CEO wants a daily pulse:

- Which features are actually used?
- Who are the top 10% power users?
- Which usage numbers look “weird” enough to investigate?

The raw file `feature_usage.csv` has **160,000 rows**. A Python loop will work. It will also be the slowest part of your job.

!!! think "Think of it like… SIMD, SQL, or Excel — not “math class.”"

    A NumPy array is a *single typed buffer*, like `int64_t usage[160000]` in C, or one column in Postgres. When you write `usage * 2`, you are not looping in Python. You are asking the CPU to map an operation over the whole column — the same idea as `SELECT usage * 2 FROM events`.

## If you already write software

A Python `for` loop over 160,000 usage events is the same smell as an N+1 query. It works in staging. It becomes the slowest line in the job.

NumPy is not “math class.” It is:

- a **columnar store** in RAM (one type, one contiguous buffer)
- a **SIMD / vectorized** API (`usage * 2` is one CPU instruction stream, not 160k Python bytecode loops)
- the thing Pandas, scikit-learn, and PyTorch all sit on

```
SQL                         NumPy
──────────────────────      ──────────────────────────
SELECT usage * 2            usage * 2
WHERE usage > 10            usage[usage > 10]
AVG(usage)                  usage.mean()
one typed column            ndarray, dtype=int64
```

### Picture the memory

A Python list of integers is a list of *pointers to objects*. Each integer is a full PyObject with a header. The CPU cannot stream that.

A NumPy array is a C array. One header, then raw values. That is why the bar chart in this lesson looks the way it does — not because NumPy is “smarter math,” but because it stops talking to the Python interpreter per row.

### What this is not

Not linear algebra. You do not need eigenvectors this week. Broadcasting is “make the shapes line up from the right, then stamp.” If you can picture a SQL `SELECT` over a column, you can picture NumPy.

## Visual: Python list vs NumPy array

A Python list of numbers is a list of **pointers to objects**. A NumPy array is a **contiguous block** of raw values.

```
Python list  [ 1,  2,  3,  4 ]
               │   │   │   │
               ▼   ▼   ▼   ▼
            int  int  int  int     ← 5 objects, extra headers, cache misses

NumPy array   [ 1 | 2 | 3 | 4 ]   ← one slab, one type, CPU-friendly
```

!!! engineer "Engineer mental model"

    `list` = JSON array of values (flexible, slow). `ndarray` = a column in a parquet file (rigid, fast). If every cell is the same type and you want the same operation on every row, use the slab.

```python
# Proof, not a slogan: loop vs vectorized multiply on 1 million numbers
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
```

## Worked example — real CloudWave usage

We will load `usage_count` as a NumPy array and ask ordinary product questions.

!!! tip "Visual cue — “one column, many questions”"

    Once the numbers live in an array, mean / median / percentile are the same *kind* of operation: walk the column once in compiled code.

```python
usage = pd.read_csv(DATA / "feature_usage.csv")
print(usage.head(3))
print(f"\n{len(usage):,} rows  |  columns: {list(usage.columns)}")

counts = usage["usage_count"].to_numpy()  # the slab
print(f"array dtype={counts.dtype}  shape={counts.shape}")

print("\nHow 'busy' is a typical event row?")
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
```

## Five lines of Pandas you need this week

Exercise 1 is a group-by. Exercise 2 is a pivot. Week 2 is the full SQL-in-Python course. This is the 5-line version so you are not surprised:

```python
by_feature = usage.groupby("feature_name")["usage_count"].sum()
print(by_feature.sort_values(ascending=False).head())

sample_ids = usage["user_id"].drop_duplicates().head(12)
grid = usage.loc[usage["user_id"].isin(sample_ids)].pivot_table(
    index="user_id", columns="feature_name",
    values="usage_count", aggfunc="sum", fill_value=0,
)
print(grid.shape)  # users × features — a 2-D slab you can hand to NumPy
```

`groupby` collapses many rows that share a key. `pivot_table` is a group-by on *two* keys, spread into a matrix. Broadcasting (next) is what you do once the matrix exists.

## Broadcasting — stamping a row down a table

Broadcasting sounds like linear algebra. It is closer to **CSS stretching** or **SQL joining a one-row lookup onto every day**.

Imagine a week of daily active users for 4 regions, and a “normal Tuesday” baseline per region:

```
DAU (7 days × 4 regions)          baseline (4 regions)
Mon  [ 80  90  70  60 ]           [ 100  100  80  70 ]
Tue  [110  95  85  72 ]      ÷
...
```

NumPy **stamps** the 4-number baseline onto every day, then divides. You never write a nested loop.

!!! math "Math, translated"

    Shapes line up *from the right*. A `(7, 4)` table and a `(4,)` row are compatible because the last dimension matches. The length-4 row is repeated 7 times. That repetition is broadcasting. It does not copy the array 7 times in your head — the library does it.

```python
# Tiny picture you can see — 7 days × 4 regions
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
```

## Z-score — “how weird is this user?”

Product people say “this account is an outlier.” Engineers should say **how many typical-spreads away from the middle**.

!!! math "Math, translated"

    `z = (value − mean) / std` → “this user is z typical-spreads from average.” `|z| > 2.5` is a useful hunt-list, not a law of nature. It is a smoke alarm, not a guilty verdict.


!!! warning "Watch out"

    On a skewed usage column the mean is dragged by whales, so z-scores flag even more whales. For “unusual,” start with **percentiles** (top 1%). Use z-scores after a log, or on something already bell-shaped.

```python
per_user = usage.groupby("user_id")["usage_count"].sum().to_numpy()
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
```

## When NumPy is the wrong tool

!!! success "Ship / don’t ship"

    - **Use NumPy** for one typed column (or a grid) and the same math on every cell: scores, percentiles, normalizations.

    - **Use Pandas** when rows have names, joins, group-bys, missing values, dates — that is next week, and it is most of SaaS work.

    - **Use SQL / Spark / DuckDB** when the table is bigger than RAM.

    - **Stay in Python** for strings, nested JSON, one-off business rules.


!!! engineer "Engineer mental model"

    Pandas is your ORM + dataframe. NumPy is the compute kernel hiding under `.values` / `.to_numpy()`. You will bounce between them all course. That is normal.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-01.md). Starter: `python exercises/ml/week-01/starter.py` from the repo root.

## 🤔 Reflection

1. A teammate writes a `for user in users` loop to compute mean usage. When do you leave it, and when do you rewrite it?
2. Why can the **mean** of CloudWave usage lie to a PM, while the **median** and **p90** tell a better story?
3. Broadcasting failed with `operands could not be broadcast`. What do you inspect first? (Print `.shape`. Align from the right.)

## Before you leave

Try one [self-check](self-checks.md#week-1-numpy) (Predict / Diagnose / Choose / Defend). Write the answer before you open the block.

## 🔗 Next week

Pandas: joins, group-bys, and building a Customer 360 table — the bread and butter of SaaS analytics. NumPy stays underneath.
