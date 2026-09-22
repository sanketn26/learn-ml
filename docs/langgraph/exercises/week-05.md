---
description: Exercises testing LangGraph idempotency, keyed charge and email writes that survive a resume without double-charging or duplicate sends.
---

# Exercises — Week 5 — Don’t Charge Twice

Do these after reading [Week 5](../week-05.md).

## Predict before you run

Charge returns 200, you *do not* write the checkpoint, process dies, resume. How many keys in `CHARGES`? What does the customer see?

## Starter / TODO

Copy `charge()` and `test_resume_does_not_double_charge` into a file. Same pattern for `send_email`.

## Runnable command

```bash
pytest your_idempotency_test.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. The table

Copy `charge()` and `test_resume_does_not_double_charge` into a file. `pytest` it.

??? tip "Hint 1 — a nudge"
    Billing's idempotency table is a dict. What's the key — and why does it include the thread *and* the invoice?

??? tip "Hint 2 — the approach"
    `charge` builds a key from `thread_id` and `invoice_id`, returns the stored result if the key exists, and writes before returning otherwise. The test calls it twice with the same state and checks one row.

??? example "Hint 3 — most of the code"
    ```python
    CHARGES: dict[str, dict] = {}


    def charge(state: dict) -> dict:
        key = f"{state['thread_id']}:charge:{state['invoice_id']}"
        if key in CHARGES:
            return {"charge": CHARGES[key], "replayed": True}
        result = {"id": key, "cents": state["cents"], "status": "captured"}
        CHARGES[key] = result
        return {"charge": result, "replayed": False}


    def test_resume_does_not_double_charge():
        state = {"thread_id": "t1", "invoice_id": "inv_9", "cents": 2900}
        first, second = charge(state), charge(state)
        assert first["charge"]["id"] == second["charge"]["id"]
        assert second["replayed"] is True
        assert len(CHARGES) == 1


    test_resume_does_not_double_charge()
    ```

## 2. Email

Same pattern for `send_email(state)`. Replaying must not append a second row to your fake `EMAILS` list.

??? tip "Hint 1 — a nudge"
    An email has no "invoice id." What makes two sends *the same* send?

??? tip "Hint 2 — the approach"
    Key it on the thread plus *what* you're sending (a template name, or the purpose like `refund-receipt`). Keep a `SENT` set of keys next to the `EMAILS` outbox; append to the outbox only when the key is new.

??? example "Hint 3 — most of the code"
    ```python
    EMAILS: list[dict] = []
    SENT: set[str] = set()


    def send_email(state: dict) -> dict:
        key = f"{state['thread_id']}:email:{state['template']}"
        # skip if the key was already sent; otherwise append to EMAILS and remember the key
        ...
    ```
    The test mirrors task 1 — write it.

## 3. Kill after success

Write the sequence: `charge` returns 200, you *do not* write the checkpoint, process dies, resume. How many keys are in `CHARGES`? What does the customer see?

??? tip "Hint 1 — a nudge"
    On resume, LangGraph reruns the node whose checkpoint never landed. So `charge` runs twice. Which side effect is protected — and by what?

??? tip "Hint 2 — the approach"
    Simulate it: call `charge(state)` (the 200), throw away its return value (the lost checkpoint), then call `charge(state)` again (the resume). Count keys and captures. Then ask what would happen if `charge` wrote the key *after* calling billing instead of before.

??? example "Hint 3 — most of the code"
    ```python
    CHARGES.clear()
    state = {"thread_id": "t9", "invoice_id": "inv_42", "cents": 4900}
    charge(state)            # billing says 200 ...
    # ... process dies here: no checkpoint written
    resumed = charge(state)  # resume reruns the node
    print(len(CHARGES), resumed["replayed"])
    ```

## 4. Cannot key it

Name one CloudWave write you would refuse to put on auto-resume (example: “tweet that we refunded them”). Put a human node in front of it, in ASCII.

??? tip "Hint 1 — a nudge"
    Idempotency needs a system on the other end that remembers keys. Which writes go somewhere that doesn't?

??? tip "Hint 2 — the approach"
    Pick a write to a system with no idempotency key (social post, SMS, a third-party webhook). Draw the graph with `interrupt_before` on that node, like week 4.

??? example "Hint 3 — a skeleton"
    ```text
    draft ──► [ human: approve? ] ──approve──► <the write you can't key>
                      │
                      └──reject──► log "cancelled"
    ```

## Expected observation

??? success "Open after you run"
    Replay does not double-charge. Replay does not append a second email. Kill-after-success still one capture if the key was written first — read the lesson if your count is 2.

## Self-check

Idempotency is a write with a key, not a prompt that says “don't charge twice.”
