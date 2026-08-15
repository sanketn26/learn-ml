# Exercises — Week 5 — Don’t Charge Twice

Do these after reading [Week 5](../week-05.md).

**1. The table.** Copy `charge()` and `test_resume_does_not_double_charge` into a file. `pytest` it.

**2. Email.** Same pattern for `send_email(state)`. Replaying must not append a second row to your fake `EMAILS` list.

**3. Kill after success.** Write the sequence: `charge` returns 200, you *do not* write the checkpoint, process dies, resume. How many keys are in `CHARGES`? (One.) What does the customer see? (One capture.)

**4. Cannot key it.** Name one CloudWave write you would refuse to put on auto-resume (example: “tweet that we refunded them”). Put a human node in front of it, in ASCII.
