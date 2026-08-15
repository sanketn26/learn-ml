# Week 3 — SQL Is the Source of Truth

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who already write `SELECT`. Read this after Week 2. The CSV in `data/` is a **fixture**. Production is a warehouse.

A model trained on a file someone emailed you is a demo. A model trained on `as of midnight, this partition` is a job.

---

## 🎯 What you will be able to do

- Treat CloudWave’s CSVs as tables you would query, not as “the data”
- Write the Customer 360 as **SQL with a date bound**
- Catch a grain bug with a test, the same way Week 2 caught an exploding join
- Know when to stay in SQL / DuckDB and when to come back to Pandas

!!! think "Think of it like… the database is git. The CSV is a checkout."

    You would not ship from a zip file on someone’s laptop. You ship from `main`. The warehouse is `main`. `as_of` is the commit you checked out. Retraining on a new CSV you cannot reproduce is training on a dirty working tree.

## If you already write software

```
Your backend                     This week
────────────────────────         ──────────────────────────────
Postgres / Snowflake             the warehouse (here: CSV + DuckDB)
dbt model                        a SELECT that has a grain
WHERE created_at < :as_of        the time-machine rule, in SQL
unit test on a fixture row       COUNT(*) after the join
ORM in the request path          Pandas after the extract
```

Week 2 built Customer 360 in Pandas. That is the ORM. This week is the query that should have produced it.

!!! warning "Watch out — the CSV is a snapshot of *all time*"

    `load_customer_360()` sums every usage row in the file, including next quarter. Fine for learning verbs. Illegal for a model you will score on Tuesday. The extract must take an **`as_of`**.

## Picture the extract

```
subscriptions          feature_usage           user_events
one row = one user     one row = user×feat×day one row = one click
        \                     |                      /
         \                    | as_of = 2024-06-01  /
          \                   | (drop later rows)  /
           \                  ▼                   /
            └────────►  customer_360_as_of  ◄────┘
                       one row = one user
                       tenure_so_far = as_of − signup
                       usage only through as_of
```

## The same 360, in SQL

DuckDB reads the files as if they were warehouse tables. The SQL is what you would schedule.

```python
import duckdb
from lib.course_data import find_data_dir

DATA = find_data_dir()
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE VIEW subscriptions AS
    SELECT * FROM read_csv_auto('{(DATA / "subscriptions.csv").as_posix()}');
    CREATE OR REPLACE VIEW feature_usage AS
    SELECT * FROM read_csv_auto('{(DATA / "feature_usage.csv").as_posix()}');
    CREATE OR REPLACE VIEW user_events AS
    SELECT * FROM read_csv_auto('{(DATA / "user_events.csv").as_posix()}');
""")

as_of = "2024-06-01"
sql_360 = f"""
WITH at_risk AS (
    SELECT user_id, plan_type, mrr, signup_date,
           datediff('day', signup_date, DATE '{as_of}') AS tenure_so_far
    FROM subscriptions
    WHERE signup_date <= DATE '{as_of}'
      AND (churn_date IS NULL OR churn_date > DATE '{as_of}')
),
usage_cut AS (
    SELECT user_id,
           SUM(usage_count) AS total_usage,
           COUNT(DISTINCT feature_name) AS features_adopted
    FROM feature_usage
    WHERE date <= DATE '{as_of}'
    GROUP BY 1
),
events_cut AS (
    SELECT user_id,
           COUNT(*) AS total_events,
           SUM(CASE WHEN event_type = 'support_message' THEN 1 ELSE 0 END) AS n_support
    FROM user_events
    WHERE timestamp <= TIMESTAMP '{as_of}'
    GROUP BY 1
)
SELECT a.user_id, a.plan_type, a.mrr, a.tenure_so_far,
       COALESCE(u.total_usage, 0) AS total_usage,
       COALESCE(u.features_adopted, 0) AS features_adopted,
       COALESCE(e.total_events, 0) AS total_events,
       COALESCE(e.n_support, 0) AS n_support
FROM at_risk a
LEFT JOIN usage_cut u USING (user_id)
LEFT JOIN events_cut e USING (user_id)
"""
frame = con.execute(sql_360).df()
print(frame.shape, frame.columns.tolist())
print(frame.head(3))
```

That query **is** `pipelines.features.build_features`. Pandas is allowed after this. Pandas is not allowed to be the only copy of the grain rules.

!!! engineer "Engineer mental model"

    One query, one grain, one `as_of`. If the warehouse team changes a column, the model job fails at extract — not three weeks later when CS notices the scores went weird. Put the SQL (or the Python that is the SQL) in git. Review it like an API.

## Grain tests are unit tests

```python
n_users = con.execute(f"""
    SELECT COUNT(*) FROM subscriptions
    WHERE signup_date <= DATE '{as_of}'
      AND (churn_date IS NULL OR churn_date > DATE '{as_of}')
""").fetchone()[0]
assert len(frame) == n_users, "360 picked up extra rows — you joined the many-side raw"
assert frame["user_id"].is_unique
assert (frame["tenure_so_far"] >= 0).all()
```

Week 2’s exploding join was a print. Here it is a red CI.

## Freshness

Ask of every extract:

1. What is the newest row I am allowed to see? (`as_of`)
2. When did this table last land? (if `max(date)` is three days old, you are scoring on a weekend of silence)
3. Can I rerun last Tuesday and get the same frame?

```python
print(con.execute("SELECT min(date), max(date) FROM feature_usage").fetchall())
print(con.execute("SELECT min(timestamp), max(timestamp) FROM user_events").fetchall())
```

CloudWave events stop at **2024-11-30**. That is the observation end of this universe. A job with `as_of=2025-01-01` is asking questions the warehouse cannot answer. Week 8 calls that **censoring**.

!!! success "Ship / don’t ship"

    Ship a model whose training table is a query plus a date. Do not ship a model whose training table is `final_final_v3.csv` on a laptop. If you cannot answer “what `as_of` built this pickle?”, you do not have a pipeline. You have a souvenir.

## ✍️ Exercise

Do the [exercises](exercises/week-03.md). The SQL lives in your head and in `pipelines/features.py`.

## 🤔 Reflection

1. Why is `tenure_days` on `subscriptions` the wrong column once you have an `as_of`?
2. A PM emails you a new CSV “with extra features.” What is your first question?
3. When would you *keep* the 360 in SQL (DuckDB, warehouse) instead of bringing it into Pandas?

## 🔗 Next

If you came from Week 2: go on to Week 4 (charts).  
If you already finished classification: Week 8 is labels, delay, and why 6.7% churn is not “just use AUC.”
