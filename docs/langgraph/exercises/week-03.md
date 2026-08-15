# Exercises — Week 3 — Persistence & Replay

Do these after reading [Week 3 — Persistence & Replay](../week-03.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Implement Persistent Workflow**

    Build a workflow with automatic checkpointing:

    - Save state after each node completes

    - Resume from latest checkpoint on restart

    - Log checkpoint history



```python
# Your implementation here!
print("Your persistent workflow here!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Build Replay System**

    Enable replay from any checkpoint:

    - List all checkpoints for a workflow

    - Resume from specific checkpoint

    - Compare results between checkpoints



```python
# Your implementation here!
print("Your replay system here!")
```

## 📝 Week 3 Project: Resilient Data Pipeline

**Build a data processing pipeline with full persistence and replay.**

### Requirements:

**Pipeline Stages:**
1. Extract (fetch data from API)
2. Validate (check data quality)
3. Transform (clean and normalize)
4. Enrich (add metadata)
5. Load (save to database)

**Persistence Features:**
- Checkpoint after each stage
- Resume from checkpoint if interrupted
- Skip already-completed stages

**Replay Capabilities:**
- List all checkpoints
- Replay from specific checkpoint
- Change transformation logic and re-run

### Test Scenarios:
1. **Complete run:** All stages succeed
2. **Failure recovery:** Fail at stage 3, resume from checkpoint
3. **Logic change:** Fix bug in Transform stage, replay
4. **What-if:** Test different Enrich strategies

```python
# Week 3 Project Starter

# TODO: Build 5-stage pipeline
# TODO: Implement persistence at each stage  
# TODO: Support resume from checkpoint
# TODO: Build replay mechanism
# TODO: Test with failure scenarios

print("🎯 Your resilient data pipeline here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Checkpointing:**
- Save state after each workflow step
- Enable fast recovery from failures

✅ **Persistence:**
- Database vs in-memory vs file storage
- Trade-offs between speed and durability

✅ **Replay:**
- Resume from checkpoints
- Replay with modified logic
- Debug and test efficiently

## 🔜 Next Week: Human-in-the-Loop

In Week 4, we'll add humans to workflows:
- Pause for human approval
- Implement feedback loops
- Build interactive workflows

---

**🎉 Congratulations on completing Week 3!** Your workflows can now survive failures and be debugged efficiently. See you next week! 🚀
