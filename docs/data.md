---
description: Reference for the synthetic CloudWave SaaS datasets used across the course, including subscriptions, events, feature usage, and feedback.
---

# CloudWave datasets

Synthetic SaaS data, checked into `data/`. One company all the way through the ML course.

| Dataset | Records | Grain | Format |
|---|---|---|---|
| [`subscriptions.csv`](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/subscriptions.csv) | 48,991 | one row per customer | CSV |
| [`user_events.csv`](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/user_events.csv) | 220,000 | one row per event | CSV |
| [`feature_usage.csv`](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feature_usage.csv) | 160,000 | one row per user × feature × day | CSV |
| [`feedback.json`](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feedback.json) | 10,000 | one object per comment | JSON Lines |
| [`product_catalog.csv`](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/product_catalog.csv) | 300 | one row per product/feature | CSV |

## subscriptions.csv

Customer lifecycle.

- `user_id` — unique customer
- `plan_type` — `free`, `starter`, `pro`, `enterprise`
- `mrr` — monthly recurring revenue in dollars (`0` for free)
- `signup_date` / `churn_date` — `churn_date` empty if still active
- `is_churned` — `1` or `0`
- `tenure_days` — signup → churn, or signup → **2024-11-30** if still active. That date is the observation end of this fixture.

## user_events.csv

Telemetry.

- `event_id`, `user_id`, `event_type` (`login`, `page_view`, `click`, `feature_use`, `payment`, `support_message`, `signup`, `upgrade`, `downgrade`, `cancel`), `timestamp`
- `device` — `web`, `ios`, `android`
- `region` — `NA`, `EMEA`, `APAC`, `LATAM`
- `session_duration` — seconds

## feature_usage.csv

Adoption.

- `user_id`, `feature_name`, `usage_count`, `avg_session_seconds`, `date`

## feedback.json

JSON Lines (one object per line), not a JSON array.

```python
import pandas as pd
feedback = pd.read_json("data/feedback.json", lines=True)
```

- `user_id`, `category`, `sentiment_score`, `feedback_text`

## How lessons load it

```python
from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
customers = load_customer_360()          # ~8k rows, laptop default
everyone = load_customer_360(n=None)     # all ~49k
```

`load_customer_360` is the Week 2 idea as a function: aggregate usage, events, and feedback to **one row per user**, then left-join onto subscriptions. Usage and events stop at **2024-11-30**. Billing rows after that date are not in the fixture.

!!! warning "Grain"
    Do not join `subscriptions` to raw `feature_usage` and then `sum(mrr)`. That number is a lie. Collapse the many-side first. Week 2 exists so you feel this once, on purpose.
