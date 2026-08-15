# Exercises — Week 1 — LangChain Fundamentals & Basic Chains

Do these after reading [Week 1 — LangChain Fundamentals & Basic Chains](../week-01.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build a Custom Triage Chain**

    Create a triage system for a **different domain**:

    - **Healthcare:** Triage patient symptoms (urgent/non-urgent)

    - **E-commerce:** Classify product reviews (return/refund/praise)

    - **HR:** Screen job applications (qualified/interview/reject)

    **Requirements:**

    - Define a Pydantic model for your domain

    - Create a prompt template with domain-specific instructions

    - Build a chain: Prompt → LLM → Parser

    - Test with 3 different inputs



```python
# Your solution here!
# Hint: Start by defining your Pydantic model

# class YourTriageModel(BaseModel):
#     category: str = Field(...)
#     priority: int = Field(...)
#     # ... add more fields

# Then create your prompt, LLM, and chain
```

!!! example "Exercise"

    **🎯 Exercise 2: Error Handling and Retries**

    Real LLM APIs can fail. Enhance the triage chain with:

    - Try/except error handling

    - Retry logic (max 3 attempts)

    - Fallback responses when LLM fails

    - Logging for debugging

    **Hint:** Use `try/except` and `time.sleep()` for exponential backoff.



```python
# Your solution here!
import time

# def triage_with_retry(ticket_input, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             result = triage_chain.invoke(ticket_input)
#             return result
#         except Exception as e:
#             # Log error, wait, retry
#             pass
#     # Return fallback
#     return {"category": "unknown", "escalate_to_human": True}
```

!!! example "Exercise"

    **🎯 Exercise 3: Multi-Step Chain**

    Build a 2-step workflow:

    - **Step 1:** Classify ticket (bug/feature/question)

    - **Step 2:** Based on classification, generate specialized response:

        Bug → Technical troubleshooting steps

        - Feature → Product roadmap information

        - Question → Knowledge base article



    **Hint:** Use `RunnableBranch` or conditional logic to route based on step 1 output.



```python
# Your solution here!
# Step 1: Classify
# Step 2: Route to specialized chain based on category
```
