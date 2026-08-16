# Exercises — Week 1 — LangGraph Fundamentals & State Management

Do these after reading [Week 1 — LangGraph Fundamentals & State Management](../week-01.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Add Error Handling**

    Enhance the document processor with robust error handling:

    - Add a `try/except` wrapper to each node

    - If a node fails, log the error to `state["errors"]`

    - Add an error recovery node that handles failures

    - Test with malformed input (empty text, None values)

    **Hint:** Create a wrapper function:

    `def safe_node(node_fn):
        def wrapper(state):
            try:
                return node_fn(state)
            except Exception as e:
                state["errors"] = state.get("errors", []) + [str(e)]
                return state
        return wrapper
    `



```python
# Your solution here!

# TODO: Create safe_node wrapper
# TODO: Wrap all nodes with error handling
# TODO: Add error recovery node
# TODO: Test with bad inputs
```

!!! example "Exercise"

    **🎯 Exercise 2: Add Conditional Routing**

    Modify the graph to skip summarization for invalid documents:

    - After `validate` node, add conditional routing

    - If `is_valid == True` → go to `summarize`

    - If `is_valid == False` → skip directly to `route`

    - Test with both valid and invalid documents

    **Hint:** Use `add_conditional_edges`:

    `def route_after_validation(state):
        return "summarize" if state["is_valid"] else "route"

    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"summarize": "summarize", "route": "route"}
    )
    `



```python
# Your solution here!

# TODO: Define routing function
# TODO: Rebuild graph with conditional edges
# TODO: Test with valid and invalid docs
```

!!! example "Exercise"

    **🎯 Exercise 3: Add Parallel Processing**

    Some operations can run in parallel. Modify the graph to:

    - After `classify`, run `extract` AND `detect_language` in parallel

    - Create a new `detect_language` node (use simple heuristics)

    - Both should finish before moving to `validate`

    **Challenge:** Research how to add parallel branches in LangGraph!



```python
# Your solution here!

# TODO: Create detect_language node
# TODO: Add parallel branches
# TODO: Test and measure performance improvement
```

## 📝 Week 1 Project: Content Moderation Pipeline

**Build a complete content moderation system for a social media platform.**

### Requirements:

**Input:** User-generated content (posts, comments)

**Workflow:**
1. **Classify** content type (text, spam, promotional, news)
2. **Detect** toxicity level (clean, mild, toxic, severe)
3. **Check** for policy violations (hate speech, misinformation, etc.)
4. **Route** based on results:
   - Clean → Auto-approve
   - Mild → Add warning label
   - Toxic → Queue for review
   - Severe → Auto-reject + alert moderators
5. **Log** all decisions for audit trail

### State Schema:
```python
class ModerationState(TypedDict):
    content_id: str
    content_text: str
    author_id: str
    
    content_type: Optional[str]
    toxicity_score: Optional[float]  # 0.0-1.0
    policy_violations: Optional[List[str]]
    
    decision: Optional[str]  # "approve", "warn", "review", "reject"
    reason: Optional[str]
    
    processed_at: Optional[str]
```

### Deliverables:
1. Complete state schema
2. 5+ node functions (classify, detect, check, route, log)
3. Graph with conditional routing
4. Test with 5 different content examples:
   - Clean post
   - Spam
   - Mild toxicity
   - Severe violation
   - Edge case (sarcasm, ambiguous)
5. ASCII diagram of your graph

### Bonus Challenges:
- Add retry logic for failed API calls
- Implement appeal mechanism (human override)
- Add metrics tracking (approval rate, false positives)
- Support multiple languages

### Starter Code:

```python
# Content Moderation Project Starter

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

# TODO: Define ModerationState
class ModerationState(TypedDict):
    pass  # Your state schema here

# TODO: Implement nodes
def classify_content(state: ModerationState) -> ModerationState:
    pass  # Your implementation

def detect_toxicity(state: ModerationState) -> ModerationState:
    pass  # Your implementation

# TODO: Build graph
# TODO: Test with examples

# Test cases
test_cases = [
    {"content_id": "1", "author_id": "user123", "content_text": "Great product! Highly recommend."},
    {"content_id": "2", "author_id": "user456", "content_text": "BUY NOW!!! 50% OFF CLICK HERE!!!"},
    {"content_id": "3", "author_id": "user789", "content_text": "This is stupid and annoying."},
    {"content_id": "4", "author_id": "user000", "content_text": "I hate you all! Worst people ever!"},
    {"content_id": "5", "author_id": "user111", "content_text": "Yeah right, like that's gonna work... 🙄"},
]
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **State Graphs > Linear Chains:**
- Shared state accessible by all nodes
- Conditional routing based on runtime values
- Support for cycles and parallel execution
- Better debugging and visualization

✅ **Core Components:**
- **State**: TypedDict with all workflow data
- **Nodes**: Functions that transform state
- **Edges**: Normal (fixed) or conditional (dynamic)
- **Entry/Exit**: START and END points

✅ **Real-World Application:**
- Built document processing pipeline
- Implemented classification, extraction, validation
- Added routing logic for different document types
- Handled errors and edge cases

✅ **Best Practices:**
- Design state schema first (types matter!)
- Keep nodes small and focused (single responsibility)
- Use conditional routing for branching logic
- Always handle errors gracefully
- Test with diverse inputs (happy path + edge cases)

## 🔜 Next Week: Complex Workflows

In Week 2, we'll build advanced workflows with:
- **Subgraphs**: Nested workflows for modularity
- **Cycles**: Retry logic and iterative refinement
- **Parallel Execution**: Speed up independent tasks
- **Dynamic Routing**: Complex multi-branch decisions
- **Real Project**: Customer onboarding system (50+ steps)

**Preview question:** How would you implement a "retry failed step up to 3 times" logic in a graph?

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [State Management Guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [Conditional Edges Examples](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [Graph Visualization Tools](https://langchain-ai.github.io/langgraph/how-tos/visualization/)

## 🐛 Debugging Tips

**Common Issues:**

1. **State not updating?**
   - Make sure nodes RETURN the updated state
   - Check for typos in state keys

2. **Conditional routing not working?**
   - Verify router function returns exact strings from edge map
   - Print state values before routing

3. **Graph hangs/infinite loop?**
   - Check that all paths eventually reach END
   - Look for cycles without exit conditions

4. **Type errors?**
   - Initialize Optional fields: `state["field"] = None`
   - Check TypedDict annotations match actual usage

---

You can now build and inspect a small state graph. The remaining weeks add the failure behavior that a durable workflow needs.
