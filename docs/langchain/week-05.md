# Week 5 — Evaluation & Debugging

**Course:** LangChain for AI Applications  
**Week Focus:** Test, evaluate, and debug LangChain applications scientifically.

---

## If you already write software

LangChain is an orchestration library, not a model. The model is the remote API (or the local weights). LangChain is the **middleware**: prompts as templates, outputs as parsers, tools as functions, memory as a store, chains as your call graph.

```
Your backend                        LangChain
─────────────────────────────       ──────────────────────────────
HTTP handler                        a chain / agent entrypoint
string template + params            PromptTemplate
JSON schema / zod / pydantic        output parser
service client                      a Tool (function + docstring)
session store / Redis               memory
try / catch + retries               callbacks, fallbacks
```

Read every abstraction as something you have already shipped. If you cannot say what the chain does as a sequence of function calls, the abstraction is hiding a bug.

## 🎯 Learning Objectives

By the end of this week, you will:
- Design and implement evaluation metrics for LLM outputs
- Use LangSmith for distributed tracing and debugging
- Build comprehensive test datasets and suites
- Handle edge cases and failure modes
- Monitor application performance in production
- Identify and fix hallucinations and errors

## 📊 Real-World Context

**The Problem:**
- Your support bot works 95% of the time
- But what about the other 5%?
- How do you know which cases fail?
- How do you fix them systematically?

**Failures to Catch:**
1. **Hallucinations:** Confident but wrong answers
2. **Wrong Tool Selection:** Using search when it should escalate
3. **Slow Responses:** Timeouts or excessive latency
4. **Missing Context:** Insufficient information retrieved
5. **Escalation Failures:** Not recognizing when to hand off to humans

**Solutions:**
- Define quantitative metrics (accuracy, relevance, latency)
- Trace execution with LangSmith
- Create golden datasets for regression testing
- Monitor production continuously
- Build feedback loops from user interactions

**Business Impact:**
- 📈 Quality: Catch issues before users do (95% → 99%+)
- 🔍 Visibility: Understand exactly what went wrong
- ⚡ Speed: Debug in minutes, not days
- 💰 Cost: Reduce wasted API calls on failures
- 📊 Continuous improvement: Data-driven optimizations


## 🔍 Part 1: Evaluation Metrics

<div class="metric-box">
<strong>Evaluation Metric:</strong> A quantitative measure of LLM application quality.
</div>

### Key Metrics for LLM Applications

| Metric | Measures | Example |
|--------|----------|----------|
| **Accuracy** | % correct answers | Support bot: Does it solve the problem? |
| **Relevance** | Info matches query | Retrieval: Are results on-topic? |
| **Latency** | Response time | API: Respond in < 2 seconds |
| **Toxicity** | Harmful language | Safety: Flag inappropriate content |
| **Hallucination** | False confidence | Fact-checking: Verify claims |
| **Tool Use** | Correct tool selection | Agents: Use right tool first try? |

### Ground Truth Datasets

You need a **golden dataset** of known good outputs:

```python
test_cases = [
    {
        "input": "How do I reset my password?",
        "expected_output": "Go to Settings > Security > Change Password",
        "expected_tool": "documentation_search",
        "max_latency_ms": 2000,
    },
    {
        "input": "I'm extremely angry about my billing",
        "expected_output": "Escalate to human agent",
        "expected_tool": "escalate_to_human",
        "max_latency_ms": 500,
    },
]
```

```python
# Implement evaluation system

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import time

class EvaluationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"

@dataclass
class TestCase:
    """Represents a single test case for evaluation."""
    input_query: str
    expected_output: str
    expected_tool: str
    max_latency_ms: int = 2000
    metadata: Dict[str, Any] = None

@dataclass
class EvaluationScore:
    """Stores evaluation results for a test run."""
    test_case: TestCase
    actual_output: str
    actual_tool: str
    latency_ms: float
    result: EvaluationResult
    accuracy_score: float  # 0-1
    relevance_score: float  # 0-1
    reasoning: str

class ApplicationEvaluator:
    """Evaluate LLM application quality systematically."""
    
    def __init__(self):
        self.test_results: List[EvaluationScore] = []
    
    def evaluate_output_match(self, expected: str, actual: str) -> float:
        """Calculate how closely actual matches expected (0-1)."""
        # Simple word overlap scoring
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        overlap = len(expected_words & actual_words)
        return overlap / len(expected_words)
    
    def evaluate_tool_selection(self, expected: str, actual: str) -> bool:
        """Check if correct tool was selected."""
        return expected.lower() == actual.lower()
    
    def evaluate_latency(self, latency_ms: float, max_ms: int) -> bool:
        """Check if latency is within SLA."""
        return latency_ms <= max_ms
    
    def run_test(self, test_case: TestCase, actual_output: str, 
                actual_tool: str, latency_ms: float) -> EvaluationScore:
        """Run a single test case evaluation."""
        
        # Score accuracy
        accuracy = self.evaluate_output_match(test_case.expected_output, actual_output)
        
        # Score tool selection
        tool_correct = self.evaluate_tool_selection(test_case.expected_tool, actual_tool)
        
        # Score latency
        latency_ok = self.evaluate_latency(latency_ms, test_case.max_latency_ms)
        
        # Determine overall result
        if tool_correct and accuracy > 0.8 and latency_ok:
            result = EvaluationResult.PASS
        elif tool_correct or accuracy > 0.6:
            result = EvaluationResult.PARTIAL
        else:
            result = EvaluationResult.FAIL
        
        reasoning = f"Accuracy: {accuracy:.1%}, Tool: {'✓' if tool_correct else '✗'}, Latency: {latency_ms:.0f}ms"
        
        score = EvaluationScore(
            test_case=test_case,
            actual_output=actual_output,
            actual_tool=actual_tool,
            latency_ms=latency_ms,
            result=result,
            accuracy_score=accuracy,
            relevance_score=1.0 if latency_ok else 0.5,
            reasoning=reasoning
        )
        
        self.test_results.append(score)
        return score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics."""
        if not self.test_results:
            return {"error": "No test results"}
        
        passes = sum(1 for r in self.test_results if r.result == EvaluationResult.PASS)
        partials = sum(1 for r in self.test_results if r.result == EvaluationResult.PARTIAL)
        fails = sum(1 for r in self.test_results if r.result == EvaluationResult.FAIL)
        
        avg_accuracy = sum(r.accuracy_score for r in self.test_results) / len(self.test_results)
        avg_latency = sum(r.latency_ms for r in self.test_results) / len(self.test_results)
        
        return {
            "total_tests": len(self.test_results),
            "passed": passes,
            "partial": partials,
            "failed": fails,
            "pass_rate": f"{100 * passes / len(self.test_results):.0f}%",
            "avg_accuracy": f"{100 * avg_accuracy:.0f}%",
            "avg_latency_ms": f"{avg_latency:.0f}ms"
        }

# Demo: Evaluation in action
print("📊 EVALUATION SYSTEM DEMO")
print("="*70)

evaluator = ApplicationEvaluator()

# Create test cases
test_cases = [
    TestCase(
        input_query="How do I reset my password?",
        expected_output="Go to Settings > Security > Change Password",
        expected_tool="documentation_search",
        max_latency_ms=2000
    ),
    TestCase(
        input_query="I'm extremely angry about billing",
        expected_output="Escalate to human support",
        expected_tool="escalate_to_human",
        max_latency_ms=500
    ),
]

# Simulate app responses
print("\n🧪 Running Test Suite...\n")

# Test 1: Good response
result1 = evaluator.run_test(
    test_cases[0],
    actual_output="Go to Settings, then Security, then Change Password",
    actual_tool="documentation_search",
    latency_ms=1250
)
print(f"Test 1: {result1.result.value.upper()}")
print(f"  {result1.reasoning}")

# Test 2: Slow escalation
result2 = evaluator.run_test(
    test_cases[1],
    actual_output="Escalating to our support team",
    actual_tool="escalate_to_human",
    latency_ms=3500  # Too slow!
)
print(f"\nTest 2: {result2.result.value.upper()}")
print(f"  {result2.reasoning}")

print(f"\n" + "="*70)
print("\n📈 SUMMARY:")
summary = evaluator.get_summary()
for key, value in summary.items():
    print(f"  {key:20} {value}")
```

## 🐛 Part 2: Debugging with Tracing

<div class="debug-box">
<strong>Tracing:</strong> Recording detailed execution path of LLM application for debugging.
</div>

### Execution Trace Example

```
User Query: "Help with my order"
  ├─ 🎯 Router: Determine category
  │  └─ LLM: "This is about orders"
  │  └─ Selected Tool: order_lookup
  │  └─ Latency: 125ms
  │
  ├─ 🔍 Order Lookup Tool
  │  └─ Query: "SELECT * FROM orders WHERE user_id=123"
  │  └─ Result: Order #ORD456 found
  │  └─ Latency: 45ms
  │
  └─ 📝 Response Generator
     └─ Template: "Your order {{order_id}} is {{status}}"
     └─ Output: "Your order ORD456 is shipped"
     └─ Latency: 87ms

Total Time: 257ms ✓
```

## 🎯 Part 3: Handling Edge Cases

### Common Failure Modes

**1. Hallucinations**
```
Input: "What's in my account?"
AI Response: "Your account has $999.99" (❌ We have no data for this user)
Fix: Validate AI output against retrieved data
```

**2. Incorrect Tool Selection**
```
Input: "I'm canceling my subscription"
AI chooses: "documentation_search" (❌ Should be "request_cancellation")
Fix: Validate tool selection matches intent
```

**3. Timeout/Slow Responses**
```
Input: "Analyze my usage data"
Response time: 8 seconds (❌ SLA is 2 seconds)
Fix: Add circuit breaker, fallback to summary
```
