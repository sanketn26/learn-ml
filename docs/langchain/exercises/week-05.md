# Exercises — Week 5 — Evaluation & Debugging

Do these after reading [Week 5 — Evaluation & Debugging](../week-05.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build Test Dataset**

    Create a comprehensive test suite:

    - Design 10 test cases covering normal + edge cases

    - Define expected outputs and SLAs

    - Track ground truth data



```python
# Exercise 1: Your test dataset here!
print("Your comprehensive test dataset implementation!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Implement Hallucination Detector**

    Detect when AI is making up answers:

    - Compare AI output against retrieved facts

    - Flag low-confidence claims

    - Escalate uncertain responses



```python
# Exercise 2: Your hallucination detector here!
print("Your hallucination detection system!")
```

## 📝 Week 5 Project: Evaluation Suite

**Build a complete evaluation and debugging system for an LLM application.**

### Requirements:

**1. Test Dataset Creation:**
- Create 50+ test cases covering:
  - Normal happy paths
  - Edge cases
  - Error conditions
  - Performance limits

**2. Evaluation Metrics:**
- Accuracy: Does output match expected?
- Relevance: Is information on-topic?
- Latency: Response time vs SLA
- Tool correctness: Right tool selected?
- Hallucination detection: Confident but wrong?

**3. Execution Tracing:**
- Log every step of execution
- Record timestamps and latencies
- Capture intermediate outputs

**4. Reporting:**
- Pass/fail summary
- Failure analysis
- Performance statistics
- Recommendations for improvement

### Deliverables:
- test_cases.json with 50+ comprehensive tests
- evaluator.py implementing metrics
- evaluation_report.md with findings
- recommendations.md with fixes

```python
# Week 5 Project Starter

# TODO: Create comprehensive test dataset (50+ cases)
# TODO: Implement evaluation metrics (accuracy, relevance, latency, hallucination)
# TODO: Build tracing system
# TODO: Generate evaluation report with findings
# TODO: Create recommendations for improvement

print("🎯 Your comprehensive evaluation suite here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Evaluation Metrics:**
- Accuracy, relevance, latency, hallucination
- How to measure LLM quality quantitatively

✅ **Test Datasets:**
- Golden datasets for regression testing
- Edge case coverage

✅ **Debugging:**
- Tracing execution paths
- Finding failure root causes

✅ **Continuous Improvement:**
- Feedback loops from testing
- Data-driven optimization

## 🔜 Next Week: Production & Deployment

In Week 6, we'll take your tested, debugged applications and deploy them at scale:
- FastAPI servers
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Load balancing and scaling
- Monitoring in production

---

You now have the beginning of a repeatable evaluation loop. It is useful only while the golden set represents failures you actually care about.
