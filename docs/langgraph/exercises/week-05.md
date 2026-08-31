# Exercises — Week 5 — Don’t Charge Twice

Do these after reading [Week 5](../week-05.md).

**1. The table.** Copy `charge()` and `test_resume_does_not_double_charge` into a file. `pytest` it.

**2. Email.** Same pattern for `send_email(state)`. Replaying must not append a second row to your fake `EMAILS` list.

**3. Kill after success.** Write the sequence: `charge` returns 200, you *do not* write the checkpoint, process dies, resume. How many keys are in `CHARGES`? (One.) What does the customer see? (One capture.)

**4. Cannot key it.** Name one CloudWave write you would refuse to put on auto-resume (example: “tweet that we refunded them”). Put a human node in front of it, in ASCII.

## Predict before you run

Charge returns 200, you *do not* write the checkpoint, process dies, resume. How many keys in `CHARGES`? What does the customer see?

## Starter / TODO

Copy `charge()` and `test_resume_does_not_double_charge` into a file. Same pattern for `send_email`.

## Runnable command

```bash
pytest your_idempotency_test.py
```

## Expected observation

Replay does not double-charge. Replay does not append a second email. Kill-after-success still one capture if the key was written first — read the lesson if your count is 2.

## Self-check

Idempotency is a write with a key, not a prompt that says “don't charge twice.”
