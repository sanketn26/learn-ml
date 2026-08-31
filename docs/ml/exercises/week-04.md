# Exercises — Week 4 — Charts That Change a Decision

## What you are building

Three charts: adoption by plan, churn by region, and an honest plan-churn bar whose title is a claim and whose y-axis starts at 0.

## Predict before you run

1. Which plan adopts the most features?
2. If the y-axis starts at 14% instead of 0, what lie does the screenshot tell?
3. Should region churn be computed on events or on users?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-04/starter.py
```

**1. Adoption curve.** For users with a `signup_date`, plot average `features_adopted` by `plan_type` as a bar. Annotate the winner.

**2. Region bars.** Most-common region per user from events, then churn rate by region. Horizontal bars, sorted.

**3. Honest title.** Rebuild the plan-churn bar so the title is a claim ("Free churn is ~2× paid") and the y-axis starts at 0.

## Success criteria

- Winner annotated on the adoption bars.
- Region chart is user-grain.
- Honest chart: title is a sentence, ylim starts at 0.

## Debugging clues

- Use the `Agg` backend in a terminal so matplotlib does not block.
- Event-level churn double-counts chatty regions.
- A truncated axis is a product bug, not a style choice.

## After you run

A chart is an API response. If the PM can misquote the title, rewrite the title.

## Lesson link

[Week 4 — Charts That Change a Decision](../week-04.md)
