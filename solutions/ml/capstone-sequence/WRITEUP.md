# Capstone — Does the Order of Events Beat the Row? — recovery writeup

Lesson: [docs/ml/capstone-sequence.md](../../../docs/ml/capstone-sequence.md)
Exercise: [docs/ml/exercises/capstone-sequence.md](../../../docs/ml/exercises/capstone-sequence.md)

!!! warning "Do not open `solution.py` until you are stuck after the hints on the exercise page"

    Work in `exercises/ml/capstone-sequence/starter.py` first.

## Reference solution

See [`solution.py`](solution.py) — about 45 seconds on a laptop CPU:

```bash
python solutions/ml/capstone-sequence/solution.py
pytest tests/test_capstone_sequence.py
```

## Why these decisions

- **Every network sees the static row.** Otherwise the comparison is "sequence vs tenure,"
  and the sequence loses for a reason that has nothing to do with order.
- **A static-only MLP is the control.** Same head, no sequence. Encoder minus control is
  what the sequence added; on this data it is negative.
- **Several seeds, spread reported.** One seed's winner changes with the seed. A verdict
  cites gaps larger than the spread.
- **Scramble, don't speculate.** Shuffling each customer's real events moves AUC by less
  than the seed spread — the models were not using order.
- **The honest verdict is "the row wins."** Median three events, event types that barely
  move the churn rate, and a `cancel` event that mostly isn't one. The architecture was
  never the bottleneck.
