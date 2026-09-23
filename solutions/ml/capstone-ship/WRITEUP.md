# Capstone — Ship the Churn Score — recovery writeup

Lesson: [docs/ml/capstone-ship.md](../../../docs/ml/capstone-ship.md)
Exercise: [docs/ml/exercises/capstone-ship.md](../../../docs/ml/exercises/capstone-ship.md)

!!! warning "Do not open `solution.py` until you are stuck after the hints on the exercise page"

    Work in `exercises/ml/capstone-ship/starter.py` first. Each step there has three
    staged hints on the exercise page.

## Reference solution

See [`solution.py`](solution.py). It runs steps 1–7 in about ten seconds on a laptop:

```bash
python solutions/ml/capstone-ship/solution.py
pytest tests/test_capstone_ship.py   # the same run, for one seed per defect
```

Step 7's reference detector flags a column when its mean moves more than 10% from last
week, or its zero share moves more than one point. Normal week-over-week movement on this
file is about 1%, so both bars sit well above the noise.

## Why these decisions

- **Horizon 90, not 30.** Thirty days leaves 49 training positives; ninety leaves 144.
  The trade is urgency for learnability, and it is written into `metrics.json`.
- **Threshold from capacity.** The 80th name's score is the threshold. It goes into
  `metrics.json` so `predict()` and the list agree about who is flagged.
- **The list is honest, not impressive.** About one real churner in 80 calls, five times a
  random list. Lift proves the model works; whether the calls are worth it is a business
  question the capstone makes you state, not answer with a metric.
- **Diagnosis over metrics.** Every seeded defect passes `validate()` and the gate. What
  catches it is a distribution check between the extract and the scorer — the test your
  postmortem should add.
