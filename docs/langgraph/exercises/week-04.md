# Exercises — Week 4 — Human-in-the-Loop & Production

Do these after reading [Week 4 — Human-in-the-Loop & Production](../week-04.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build Simple Approval System**

    Create a workflow with basic approval gate:

    - Submit request for approval

    - Human approves/rejects

    - Route to appropriate next step



```python
# Exercise 1: Your simple approval system here!
print("Your approval system implementation here!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Implement SLA Monitoring**

    Monitor approval SLAs:

    - Track pending approvals

    - Alert if SLA breached

    - Auto-escalate stale requests



```python
# Exercise 2: Your SLA monitoring here!
print("Your SLA monitoring implementation here!")
```

## 📝 Week 4 Project: Expense Report Approval System

**Build a complete human-in-the-loop workflow for expense reports.**

### Requirements:

**Workflow Stages:**
1. **Submit:** Employee submits expenses with receipts
2. **Validate:** AI checks policy compliance
3. **Review:** Manager approves/rejects
4. **Escalate:** High-value or suspicious go to director
5. **Process:** Approved expenses → payment
6. **Notify:** Inform employee of outcome

**Human-in-the-Loop Features:**
- Managers can approve, reject, or request clarification
- Auto-escalate: >$5000 or risk_score>0.7
- SLA: Manager approval within 48 hours
- Appeal: Employee can ask for reconsideration

**Metrics to Track:**
- Approval time (average, p95)
- Approval rate (% approved)
- Escalation rate
- Appeal rate

```python
# Week 4 Project Starter

# TODO: Build expense report submission
# TODO: AI validation of policy compliance
# TODO: Manager review workflow
# TODO: Auto-escalation rules
# TODO: Track SLA metrics
# TODO: Test various approval paths

print("🎯 Your expense report approval system here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Approval Gates:**
- Pause workflows for human review
- Prevent automated errors in critical paths

✅ **Human Feedback:**
- Integrate human input into workflow logic
- Escalation rules for complex decisions

✅ **Conditional Routing:**
- Different paths for approve/reject/needs-info
- Dynamic workflow behavior

✅ **Production Deployment:**
- SLA management
- Metrics and monitoring
- Scalable approval infrastructure

## 🚀 Final Capstone Challenge

**Build an AI-Powered Loan Approval System** combining all 4 weeks:

1. **Week 1:** State graphs define approval workflow
2. **Week 2:** Conditional routing based on loan amount
3. **Week 3:** Persist application state, allow resumption
4. **Week 4:** Human underwriters approve/deny with feedback

---

**🎉 Congratulations on completing LangGraph!** You can now build sophisticated, resilient, human-centered AI workflows. See you in the next course! 🚀
