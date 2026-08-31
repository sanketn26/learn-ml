# Week 04 — recovery writeup

Lesson: [docs/ml/week-04.md](../../../docs/ml/week-04.md)
Exercise: [docs/ml/exercises/week-04.md](../../../docs/ml/exercises/week-04.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-04/starter.py` first.

## Hint 1

??? tip "Hint 1"

    A chart is an API response: title is the claim, axis starts at a
    honest zero, the winner is annotated. Region churn needs one region per
    user *before* you group. Do not make the reader do the subtraction in
    their head.

## Hint 2

??? tip "Hint 2"

    `load_customer_360()` already has `features_adopted`. Bar by `plan_type`.
    For regions, collapse `user_events` with a mode, left-join, then
    `groupby("region")["is_churned"].mean()` and `barh`. Set `ax.set_ylim(0, ...)`.

## Debugging clues

??? warning "Debugging clues"

    - A truncated y-axis turns a 1-point gap into a crisis.
    - Plotting event-level churn (before collapsing users) double-counts
      chatty regions.
    - `matplotlib` in a terminal: use the `Agg` backend so the script does
      not block on a window.
    - Annotate with the *value*, not just a color.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-04/solution.py
```

The script prints the numbers and writes PNG files next to itself. The
honest-title chart is the one whose y-axis starts at 0.

## Why this decision

Free vs paid churn in this file is roughly 2×. That claim belongs in the
title so a PM can screenshot it without rewriting it. Starting the axis at
zero is the difference between "the bar is twice as tall" and "the bar
starts at 14% so everything looks dramatic."
