---
description: Exercises building a LangGraph human-in-the-loop approval graph with interrupt_before, testing approve, reject, and needs-info resume paths.
---

# Exercises — Week 4 — Approve / reject / needs-info

Do these after reading [Week 4](../week-04.md). One small approval graph. Weeks 1–4; week 5 (idempotency) is next — mention it, do not skip it later.

```python
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["approve"])
# ...
app.invoke(payload, config)          # pauses
app.update_state(config, {"decision": ...})
app.invoke(None, config)             # continues
```

## 1. Pause

CloudWave refund: `draft` then `approve`. Compile with `interrupt_before=["approve"]`.

**Checks:**

- After the first `invoke`, `get_state(config).next` includes `approve`
- `log` has the draft line and **not** `executed`

## 2. Three paths

Separate `thread_id`s for `approve`, `reject`, `needs_info`.

**Checks:**

- approve → last log `executed`
- reject → last log `cancelled`
- needs-info → last log `asked-for-info` (or your equivalent)

## 3. Weeks 1–4, not a loan platform

In five lines, list which week each piece is (branch / reducer, optional extra node, MemorySaver, interrupt). Add one sentence: the write is still at-least-once until week 5 keys it.

**Checks:**

- No `ApprovalRequest` class
- No loan-underwriting project

## Predict before you run

After the first `invoke` with `interrupt_before=["approve"]`, does `get_state(config).next` include `approve`? Has `executed` already been logged?

## Runnable command

```bash
python your_refund_approval.py
```

## Expected observation

Pause before approve. Three `thread_id`s: executed / cancelled / asked-for-info. No `ApprovalRequest` class.

## Self-check

Weeks 1–4 pieces named. The write is still at-least-once until week 5 keys it.
