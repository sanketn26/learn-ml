# Exercises — Week 2 — Memory & Conversation Management

Do these after reading [Week 2 — Memory & Conversation Management](../week-02.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build a Customer Support Bot**

    Create a customer support chatbot that:

    - Remembers the customer's issue throughout the conversation

    - Tracks troubleshooting steps already tried

    - Escalates to human if issue not resolved after 5 turns

    - Uses ConversationBufferWindowMemory (k=5)

    **Test scenarios:**

    - Password reset request

    - Bug report (export not working)

    - Billing question



```python
# Your solution here!
from langchain.memory import ConversationBufferWindowMemory

# TODO:
# 1. Create memory with k=5
# 2. Build support prompt with troubleshooting steps
# 3. Track turn count
# 4. Escalate if turn_count > 5

# support_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
# turn_count = 0

# def support_chat(user_input):
#     global turn_count
#     turn_count += 1
#     
#     if turn_count > 5:
#         return "I'm escalating this to our specialist team. You'll hear from us within 1 hour."
#     
#     # Your chain logic here
#     pass
```

!!! example "Exercise"

    **🎯 Exercise 2: Compare Memory Types**

    Test the same 10-turn conversation with different memory types:

    - ConversationBufferMemory

    - ConversationBufferWindowMemory (k=3)

    - ConversationSummaryMemory

    **Compare:**

    - Token usage (approximate by character count)

    - Context retention (does it remember turn 1 at turn 10?)

    - Response quality

    **Conversation topic:** Planning a vacation (destination, dates, activities, budget)



```python
# Your solution here!

# Test conversation:
vacation_conversation = [
    "I want to plan a vacation",
    "Somewhere tropical with beaches",
    "Budget is around $3000",
    "I like snorkeling and hiking",
    "Prefer July or August",
    "How's the weather in Bali during July?",
    "What about Maldives?",
    "I need a good hotel recommendation",
    "Remember, I said I like snorkeling",
    "And my budget was $3000 - does this fit?"
]

# TODO: Test with each memory type and compare results
```

!!! example "Exercise"

    **🎯 Exercise 3: Multi-User Chat Room**

    Build a simple chat room where:

    - 3 users can chat simultaneously

    - Each user has their own conversation memory

    - Assistant responds contextually to each user

    - Simulate 5 turns per user (15 total messages)

    **Users:**

    - Alice: Shopping for laptops

    - Bob: Looking for running shoes

    - Charlie: Asking about return policy



```python
# Your solution here!

# users = {
#     "alice": {"memory": ConversationBufferMemory(return_messages=True)},
#     "bob": {"memory": ConversationBufferMemory(return_messages=True)},
#     "charlie": {"memory": ConversationBufferMemory(return_messages=True)}
# }

# def chat_room(user_id, message):
#     memory = users[user_id]["memory"]
#     # Process message with user's memory
#     pass
```
