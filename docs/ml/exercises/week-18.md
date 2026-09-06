---
description: Tune a 1-D convolutional neural network's kernel size, compare it against a dense-flatten baseline, and sketch how its sliding stencil detects patterns.
---

# Exercises — Week 18 — CNNs: Sliding Detectors

## What you are building

A 1-D CNN with a wider kernel, a dense-flatten baseline, and a stencil sketch on 12 week-boxes.

## Predict before you run

1. Does `kernel_size=5` move test accuracy, or just change what the detector looks at?
2. If a `Linear(12, 1)` ties the CNN, was the signal *shape* or *total*?
3. Which of three stencil positions fires on a late-week drop?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-18/starter.py
```

**1. Kernel size.** Change `kernel_size` to 5. Does test accuracy move? What did you make the detector look at?

**2. Dense baseline.** Flatten the 12 weeks into a `nn.Linear(12, 1)` and compare. If they tie, the *shape* was not the signal — the *total* was.

**3. Draw it.** Sketch one user as 12 boxes and a 3-wide stencil in three positions. Circle the position you think fires on a late-week drop.

## Success criteria

- Kernel-5 vs kernel-3 note.
- Dense baseline AUC/accuracy next to the CNN.
- ASCII or paper sketch with a circled position.

## Debugging clues

- This is a teaching toy on 12 weekly totals. CPU only.
- `load_weekly_usage_grid` is not an as-of label.
- A tie with the dense net means stop claiming “the CNN saw the drop-off.”

## After you run

A convolution is one detector, many places. On CloudWave's 7-column table, last week's GBT still ships.

## Lesson link

[Week 18 — CNNs: Sliding Detectors](../week-18.md)
