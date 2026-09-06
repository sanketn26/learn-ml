---
description: Make LangGraph resume idempotent by keying side effects like charges and emails so an at-least-once retry never double-executes a write.
---

# Week 5 — Crash, Resume, Don’t Charge Twice

**Course:** LangGraph  
**Who this is for:** Engineers who have a graph (Weeks 1–4) that talks to billing. Persistence without **idempotency** is a double-charge machine.

---

## 🎯 What you will be able to do

- Name the side effects in a CloudWave refund / pause-subscription flow
- Resume after node 3 without running node 3 again
- Put a **write token** (idempotency key) on every tool that moves money
- Test “kill the process after `charge`, start again, still one charge”

!!! think "Think of it like… a payment webhook."

    Stripe asks for an idempotency key because networks retry. Your graph retries too — on purpose (Week 3 checkpoints). If `charge()` is not idempotent, resume is a second capture. The runtime owning state does not make your bank API owned.

## If you already write software

```
At-least-once delivery     the graph will run a node ≥ 1 time
Exactly-once *effect*      your job, with a key
Checkpoint                 “we finished parse and retrieve”
Resume                     start at charge
Bad                        charge() again because the process died after the HTTP 200
Good                       charge(key=thread_id+node+payload_hash) → same result
```

### Picture the failure

```
parse → retrieve → charge → email
                     │
                     ● process dies after 200, before checkpoint flush
                     │
                  resume
                     │
                     ▼
                   charge again   ← unless the key hits the same row
```

Week 3’s checkpoint is necessary and not sufficient.

```python
CHARGES: dict[str, dict] = {}  # stand-in for billing's idempotency table


def charge(state: dict) -> dict:
    key = f"{state['thread_id']}:charge:{state['invoice_id']}"
    if key in CHARGES:
        return {"charge": CHARGES[key], "replayed": True}
    result = {"id": key, "cents": state["cents"], "status": "captured"}
    CHARGES[key] = result
    return {"charge": result, "replayed": False}
```

The test is the lesson:

```python
def test_resume_does_not_double_charge():
    state = {"thread_id": "t1", "invoice_id": "inv_9", "cents": 2900}
    first = charge(state)
    second = charge(state)          # replay after a crash
    assert first["charge"]["id"] == second["charge"]["id"]
    assert second["replayed"] is True
    assert len(CHARGES) == 1
```

Email is the same shape (`email_id` = `thread_id + template + user`). Some side effects you **cannot** make idempotent (a webhook to a vendor who ignores keys). Those nodes need a human (Week 4) or an outbox table you reconcile.

!!! warning "Watch out"

    - Retrying `email` is how customers get three “we refunded you” mails. Key it, or make it the last node and accept at-least-once with a template that is safe to repeat.
    - `time.sleep` + retry is not idempotency.
    - A subgraph that “just calls the parent’s tools” inherits this problem.

!!! success "Ship / don’t ship"

    Ship a graph whose every write has a key you can show in a table. Do not ship resume on a refund path until `test_resume_does_not_double_charge` exists. Do not tell the auditor “LangGraph checkpoints, so we’re exactly-once.”

## ✍️ Exercise

[Exercises](exercises/week-05.md).

## 🤔 Reflection

1. List the writes in your CloudWave “cancel + refund + email” graph. Which have keys today?
2. The checkpoint flushed *before* the HTTP 200 came back. What happens on resume? What should `charge()` do?
3. Why is a human-in-the-loop node the right answer for a write you cannot key?

## 🔗 After

You have the three reasons to use a graph: branch, wait, resume. If you only needed a sequence of reads, go back to a function.
