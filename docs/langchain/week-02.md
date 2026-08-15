# Week 2 — Memory & Conversation Management

**Course:** LangChain for AI Applications  
**Week Focus:** Build stateful conversational AI that remembers context, manages long dialogues, and provides personalized experiences.

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
- Understand different memory types and their trade-offs
- Implement conversation buffers, windows, and summaries
- Build multi-turn dialogue systems with context retention
- Handle long conversations without hitting token limits
- Manage multiple user sessions simultaneously
- Create personalized shopping assistants that remember preferences

## 📊 Real-World Context

**The Challenge:** Your e-commerce platform processes 50,000 customer conversations daily:
- 60% are product discovery ("show me blue dresses under $100")
- 25% are order tracking and modifications
- 10% are returns and complaints
- 5% are complex multi-step purchases

**Problems with stateless chatbots:**
- ❌ Customer: "Show me laptops" → Bot: "Here are 10 laptops"
- ❌ Customer: "The second one looks good" → Bot: "I don't know what you're referring to"
- ❌ Customer gets frustrated and abandons cart

**The Solution:** A shopping assistant with memory that:
1. **Remembers** previous messages and product views
2. **Tracks** user preferences (size, color, budget)
3. **Maintains** shopping cart context
4. **Personalizes** recommendations based on conversation history
5. **Summarizes** long conversations to save tokens

**Business Impact:**
- 🛒 Increase conversion rate from 2.3% → 6.8% (3x improvement)
- 💰 Add $2.4M in annual revenue
- ⏱️ Reduce avg conversation time from 8 min → 4 min
- 😊 Increase CSAT score from 3.2 → 4.5/5

Companies like **Amazon, Shopify, and Zalando** use conversational AI with memory to drive billions in sales.


## 🔍 Part 1: The Problem — Why Memory Matters

### Stateless vs Stateful Conversations

```python
from langchain.llms.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ❌ Problem: Stateless chatbot (no memory)
print("❌ STATELESS CHATBOT (No Memory)\n" + "="*50)

stateless_prompt = ChatPromptTemplate.from_template(
    "You are a shopping assistant. User says: {input}"
)

stateless_llm = FakeListLLM(responses=[
    "Here are our top laptops: 1) Dell XPS 13 ($999), 2) MacBook Air ($1099), 3) ThinkPad X1 ($1299)",
    "I don't have context about which laptop you're referring to. Could you please specify?",
    "I don't see any cart information. What would you like to add?"
])

stateless_chain = stateless_prompt | stateless_llm | StrOutputParser()

# Simulate conversation
msg1 = stateless_chain.invoke({"input": "Show me laptops under $1000"})
print(f"User: Show me laptops under $1000")
print(f"Bot: {msg1}\n")

msg2 = stateless_chain.invoke({"input": "Tell me more about the first one"})
print(f"User: Tell me more about the first one")
print(f"Bot: {msg2}\n")

msg3 = stateless_chain.invoke({"input": "Add it to my cart"})
print(f"User: Add it to my cart")
print(f"Bot: {msg3}\n")

print("⚠️ Problem: Bot has no context! Every message is treated independently.")
```

### The Solution: Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# ✅ Solution: Stateful chatbot (with memory)
print("✅ STATEFUL CHATBOT (With Memory)\n" + "="*50)

# Create memory
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# Create prompt with memory placeholder
stateful_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful shopping assistant. Remember the conversation context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

stateful_llm = FakeListLLM(responses=[
    "Here are our top laptops under $1000: 1) Dell XPS 13 ($999) - 13.3 inch, 16GB RAM, perfect for travel, 2) HP Pavilion ($849) - 15.6 inch, gaming capable, 3) Lenovo IdeaPad ($749) - budget-friendly, great battery life",
    "The Dell XPS 13 ($999) is an excellent choice! It features: Intel i7 processor, 16GB RAM, 512GB SSD, 13.3-inch 4K display, weighs only 2.6 lbs, 12-hour battery life. It's perfect for professionals and students. Would you like to add it to your cart?",
    "Great! I've added the Dell XPS 13 ($999) to your cart. Your cart total is $999. Would you like to continue shopping or proceed to checkout?"
])

stateful_chain = stateful_prompt | stateful_llm | StrOutputParser()

# Helper function to run chain with memory
def chat_with_memory(user_input, chain, memory):
    # Load conversation history
    history = memory.load_memory_variables({})["history"]
    
    # Invoke chain
    response = chain.invoke({
        "input": user_input,
        "history": history
    })
    
    # Save to memory
    memory.save_context({"input": user_input}, {"output": response})
    
    return response

# Simulate conversation
msg1 = chat_with_memory("Show me laptops under $1000", stateful_chain, memory)
print(f"User: Show me laptops under $1000")
print(f"Bot: {msg1}\n")

msg2 = chat_with_memory("Tell me more about the first one", stateful_chain, memory)
print(f"User: Tell me more about the first one")
print(f"Bot: {msg2}\n")

msg3 = chat_with_memory("Add it to my cart", stateful_chain, memory)
print(f"User: Add it to my cart")
print(f"Bot: {msg3}\n")

print("✅ Success: Bot maintains context throughout the conversation!")
```

## 📚 Part 2: Memory Types Deep Dive

LangChain provides multiple memory types, each with different trade-offs:

| Memory Type | Stores | Token Usage | Best For |
|-------------|--------|-------------|----------|
| **ConversationBufferMemory** | All messages | High | Short conversations |
| **ConversationBufferWindowMemory** | Last N messages | Medium | Fixed-length dialogues |
| **ConversationSummaryMemory** | Summary of conversation | Low | Long conversations |
| **ConversationSummaryBufferMemory** | Summary + recent messages | Medium | Hybrid approach |

### 2.1 ConversationBufferMemory — Store Everything

```python
from langchain.memory import ConversationBufferMemory

# Stores complete conversation history
buffer_memory = ConversationBufferMemory()

# Simulate conversation
buffer_memory.save_context(
    {"input": "I'm looking for running shoes"},
    {"output": "Great! What's your budget and preferred brand?"}
)
buffer_memory.save_context(
    {"input": "Under $150, prefer Nike or Adidas"},
    {"output": "Perfect! Here are 3 options: Nike Pegasus ($140), Adidas Ultraboost ($145), Nike React ($130)"}
)
buffer_memory.save_context(
    {"input": "Tell me about the Pegasus"},
    {"output": "Nike Pegasus ($140) is our bestseller: cushioned sole, breathable mesh, 300-mile durability. Available in sizes 7-13. Popular for daily training and races."}
)

print("🗄️ ConversationBufferMemory — Stores ALL messages\n")
print(buffer_memory.load_memory_variables({}))
print("\n✅ Pros: Perfect recall, no information loss")
print("❌ Cons: Token usage grows linearly, expensive for long conversations")
print("💡 Use case: Short conversations (< 20 messages), customer support chats")
```

### 2.2 ConversationBufferWindowMemory — Keep Last N Messages

```python
from langchain.memory import ConversationBufferWindowMemory

# Only keeps last 2 interactions (4 messages: 2 human + 2 AI)
window_memory = ConversationBufferWindowMemory(k=2)

# Add 4 interactions
window_memory.save_context(
    {"input": "Show me laptops"},
    {"output": "Here are 5 laptops..."}
)
window_memory.save_context(
    {"input": "Under $1000 please"},
    {"output": "Filtered to 3 laptops under $1000..."}
)
window_memory.save_context(
    {"input": "Do you have any in blue?"},
    {"output": "Yes! The Dell XPS comes in blue..."}
)
window_memory.save_context(
    {"input": "Add it to cart"},
    {"output": "Added Dell XPS (blue) to your cart!"}
)

print("🪟 ConversationBufferWindowMemory — Keeps last K=2 interactions\n")
print(window_memory.load_memory_variables({}))
print("\n✅ Pros: Fixed token usage, prevents context overflow")
print("❌ Cons: Forgets older context (e.g., forgot 'laptops under $1000')")
print("💡 Use case: Chatbots with turn-based context (games, forms)")
```

### 2.3 ConversationSummaryMemory — Summarize History

```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms.fake import FakeListLLM

# Uses LLM to create a running summary of the conversation
summary_llm = FakeListLLM(responses=[
    "User is shopping for laptops under $1000.",
    "User is shopping for laptops under $1000. User prefers Dell brand and blue color.",
    "User is shopping for laptops under $1000. User prefers Dell brand and blue color. User added Dell XPS (blue, $999) to cart."
])

summary_memory = ConversationSummaryMemory(llm=summary_llm)

# Add interactions
summary_memory.save_context(
    {"input": "Show me laptops under $1000"},
    {"output": "Here are 3 laptops: Dell XPS ($999), HP Pavilion ($849), Lenovo IdeaPad ($749)"}
)
summary_memory.save_context(
    {"input": "Do you have the Dell in blue?"},
    {"output": "Yes! The Dell XPS is available in blue for $999."}
)
summary_memory.save_context(
    {"input": "Add it to my cart"},
    {"output": "Added Dell XPS (blue, $999) to your cart!"}
)

print("📝 ConversationSummaryMemory — Summarizes conversation\n")
print(summary_memory.load_memory_variables({}))
print("\n✅ Pros: Constant token usage, scales to very long conversations")
print("❌ Cons: Information loss, extra LLM call for summarization")
print("💡 Use case: Long support sessions, therapy chatbots, multi-day conversations")
```

### 2.4 ConversationSummaryBufferMemory — Best of Both Worlds

```python
from langchain.memory import ConversationSummaryBufferMemory

# Keeps summary + recent messages (when total tokens < max_token_limit)
hybrid_llm = FakeListLLM(responses=[
    "User requested laptops under $1000. Showed Dell XPS ($999), HP Pavilion ($849), Lenovo IdeaPad ($749)."
])

hybrid_memory = ConversationSummaryBufferMemory(
    llm=hybrid_llm,
    max_token_limit=100  # When exceeded, older messages get summarized
)

# Add messages
hybrid_memory.save_context(
    {"input": "Show me laptops under $1000"},
    {"output": "Here are 3 options: Dell XPS ($999), HP Pavilion ($849), Lenovo IdeaPad ($749)"}
)
hybrid_memory.save_context(
    {"input": "What's the battery life on the Dell?"},
    {"output": "The Dell XPS has 12-hour battery life."}
)
hybrid_memory.save_context(
    {"input": "Perfect, add it to cart"},
    {"output": "Added Dell XPS ($999) to cart. Total: $999"}
)

print("🔀 ConversationSummaryBufferMemory — Hybrid approach\n")
print(hybrid_memory.load_memory_variables({}))
print("\n✅ Pros: Balanced token usage, retains recent context + summary of old")
print("❌ Cons: More complex, requires tuning max_token_limit")
print("💡 Use case: Production chatbots, default choice for most applications")
```

## 🛠️ Part 3: Building a Shopping Assistant

<div class="scenario-box">
<strong>📌 Scenario:</strong> Build an e-commerce shopping assistant that:
<ol>
<li>Remembers user preferences (size, color, budget)</li>
<li>Tracks products viewed and added to cart</li>
<li>Provides personalized recommendations</li>
<li>Handles multi-step checkout process</li>
<li>Manages conversations across sessions</li>
</ol>
</div>

### Step 1: Define Product Catalog

```python
# Product database
PRODUCTS = {
    "laptops": [
        {"id": "L001", "name": "Dell XPS 13", "price": 999, "specs": "13.3in, i7, 16GB RAM, 512GB SSD", "colors": ["silver", "black"]},
        {"id": "L002", "name": "MacBook Air", "price": 1099, "specs": "13.6in, M2, 8GB RAM, 256GB SSD", "colors": ["silver", "gold", "space gray"]},
        {"id": "L003", "name": "ThinkPad X1", "price": 1299, "specs": "14in, i7, 16GB RAM, 1TB SSD", "colors": ["black"]},
        {"id": "L004", "name": "HP Pavilion", "price": 849, "specs": "15.6in, i5, 8GB RAM, 512GB SSD", "colors": ["silver", "blue"]},
    ],
    "shoes": [
        {"id": "S001", "name": "Nike Pegasus", "price": 140, "type": "running", "sizes": [7, 8, 9, 10, 11, 12]},
        {"id": "S002", "name": "Adidas Ultraboost", "price": 145, "type": "running", "sizes": [7, 8, 9, 10, 11, 12, 13]},
        {"id": "S003", "name": "Nike React", "price": 130, "type": "training", "sizes": [8, 9, 10, 11]},
    ]
}

# Shopping cart
cart = []

print("📦 Product Catalog Loaded")
print(f"  Laptops: {len(PRODUCTS['laptops'])} models")
print(f"  Shoes: {len(PRODUCTS['shoes'])} models")
```

### Step 2: Build Shopping Assistant with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.llms.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser

# Create memory for shopping session
shopping_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Create prompt with conversation history
shopping_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful e-commerce shopping assistant.
    
    Available products:
    LAPTOPS: Dell XPS 13 ($999), MacBook Air ($1099), ThinkPad X1 ($1299), HP Pavilion ($849)
    SHOES: Nike Pegasus ($140), Adidas Ultraboost ($145), Nike React ($130)
    
    Remember user preferences and provide personalized recommendations.
    Keep track of items in cart and help with checkout."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create LLM with realistic responses
shopping_llm = FakeListLLM(responses=[
    "Great! I can help you find the perfect laptop. What's your budget and main use case (work, gaming, student, etc.)?",
    "For work under $1000, I recommend the Dell XPS 13 ($999) - it's our bestseller! It has a 13.3-inch display, Intel i7 processor, 16GB RAM, and 512GB SSD. Perfect for multitasking and productivity. Available in silver or black. Would you like to see detailed specs?",
    "The Dell XPS 13 comes in silver and black. Which color do you prefer?",
    "Excellent choice! I've added the Dell XPS 13 (silver, $999) to your cart. Your cart total is $999. Would you like to continue shopping or proceed to checkout?",
    "Perfect timing! We also have running shoes on sale. The Nike Pegasus ($140) would pair great with your new laptop for staying active. Would you like to see our shoe collection?",
    "Here are our top running shoes:\n1. Nike Pegasus ($140) - Cushioned, 300-mile durability, sizes 7-12\n2. Adidas Ultraboost ($145) - Premium comfort, energy return, sizes 7-13\n3. Nike React ($130) - Responsive, training-focused, sizes 8-11\n\nWhat's your shoe size?",
    "Great! The Nike Pegasus is available in size 10. I've added it to your cart. Your cart now has:\n1. Dell XPS 13 (silver) - $999\n2. Nike Pegasus (size 10) - $140\n\nCart total: $1,139. Ready to checkout?",
    "Wonderful! Proceeding to checkout with 2 items totaling $1,139. You'll receive free shipping on orders over $1000. Would you like me to apply any promo codes or gift cards?"
])

shopping_chain = shopping_prompt | shopping_llm | StrOutputParser()

# Helper function for shopping conversation
def shop(user_input):
    """Process user input and return assistant response."""
    history = shopping_memory.load_memory_variables({})["chat_history"]
    response = shopping_chain.invoke({
        "input": user_input,
        "chat_history": history
    })
    shopping_memory.save_context({"input": user_input}, {"output": response})
    return response

print("🛒 Shopping Assistant Ready!")
```

### Step 3: Interactive Shopping Session

```python
# Simulate a complete shopping session
print("🛍️ SHOPPING SESSION\n" + "="*60)

conversation = [
    "I need a laptop for work",
    "Under $1000, mainly for emails and video calls",
    "What colors does it come in?",
    "I'll take the silver one",
    "What else do you recommend?",
    "Show me running shoes",
    "Size 10, I'll take the Pegasus",
    "Let's checkout"
]

for user_msg in conversation:
    print(f"\n👤 Customer: {user_msg}")
    bot_response = shop(user_msg)
    print(f"🤖 Assistant: {bot_response}")
    print("-" * 60)

print("\n✅ Shopping session completed successfully!")
print("\n📊 CONVERSATION ANALYSIS:")
print(f"  Total interactions: {len(conversation)}")
print(f"  Products discussed: Laptops, Running shoes")
print(f"  Items in cart: 2 (Dell XPS 13, Nike Pegasus)")
print(f"  Cart total: $1,139")
print(f"  Memory preserved: ✅ Bot remembered context throughout")
```

## 🔧 Part 4: Advanced Memory Patterns

### 4.1 Multi-User Session Management

```python
from langchain.memory import ConversationBufferMemory

# Store separate memory for each user
user_sessions = {}

def get_user_memory(user_id):
    """Get or create memory for a specific user."""
    if user_id not in user_sessions:
        user_sessions[user_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
    return user_sessions[user_id]

# Simulate multiple users shopping simultaneously
print("👥 MULTI-USER SESSION MANAGEMENT\n" + "="*60)

# User A: Shopping for laptops
memory_a = get_user_memory("user_alice")
memory_a.save_context(
    {"input": "Show me gaming laptops"},
    {"output": "Here are top gaming laptops: HP Pavilion ($849) with dedicated GPU..."}
)

# User B: Shopping for shoes
memory_b = get_user_memory("user_bob")
memory_b.save_context(
    {"input": "I need running shoes size 11"},
    {"output": "Perfect! Here are size 11 running shoes: Nike Pegasus ($140), Adidas Ultraboost ($145)..."}
)

# User A continues (context preserved)
memory_a.save_context(
    {"input": "What's the graphics card on that HP?"},
    {"output": "The HP Pavilion has NVIDIA GTX 1650 graphics card, perfect for gaming and video editing."}
)

print("👤 User Alice's conversation:")
print(memory_a.load_memory_variables({})['history'])
print("\n👤 User Bob's conversation:")
print(memory_b.load_memory_variables({})['history'])

print("\n✅ Each user has isolated conversation history!")
print(f"Active sessions: {len(user_sessions)}")
```

### 4.2 Persistent Memory (Save/Load Sessions)

```python
import json
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Save conversation to disk
def save_conversation(memory, user_id):
    """Save conversation history to JSON file."""
    messages = memory.load_memory_variables({})["history"]
    
    # Convert messages to serializable format
    serialized = []
    for msg in messages:
        serialized.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content
        })
    
    conversation_data = {
        "user_id": user_id,
        "messages": serialized,
        "timestamp": "2025-11-09T10:30:00"
    }
    
    print(f"💾 Saving conversation for {user_id}:")
    print(json.dumps(conversation_data, indent=2))
    return conversation_data

# Load conversation from disk
def load_conversation(conversation_data):
    """Load conversation history from JSON."""
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    
    messages = conversation_data["messages"]
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            human_msg = messages[i]["content"]
            ai_msg = messages[i + 1]["content"]
            memory.save_context({"input": human_msg}, {"output": ai_msg})
    
    print(f"\n📂 Loaded conversation for {conversation_data['user_id']}")
    return memory

# Demo: Save and restore session
print("💾 PERSISTENT MEMORY DEMO\n" + "="*60)

# Create a conversation
session_memory = ConversationBufferMemory(return_messages=True, memory_key="history")
session_memory.save_context(
    {"input": "I want a laptop under $1000"},
    {"output": "Great! I recommend the Dell XPS 13 ($999) or HP Pavilion ($849)."}
)
session_memory.save_context(
    {"input": "Tell me about the Dell"},
    {"output": "The Dell XPS 13 features: 13.3in display, i7 processor, 16GB RAM, 512GB SSD. Excellent for work!"}
)

# Save to "disk"
saved_data = save_conversation(session_memory, "user_charlie")

# Later: Restore from "disk"
restored_memory = load_conversation(saved_data)

# Continue conversation
restored_memory.save_context(
    {"input": "I'll take the Dell!"},
    {"output": "Excellent choice! Added Dell XPS 13 to your cart."}
)

print("\n✅ Conversation restored and continued successfully!")
print("\n💡 Use case: Resume conversations across sessions, store in database")
```

### 4.3 Selective Memory — Remember Important Facts

```python
from langchain.memory import ConversationBufferMemory

# Store user preferences separately from conversation
class ShoppingPreferences:
    def __init__(self):
        self.budget = None
        self.preferred_brands = []
        self.sizes = {}
        self.colors = []
        self.categories = []
    
    def update(self, key, value):
        setattr(self, key, value)
    
    def to_dict(self):
        return {
            "budget": self.budget,
            "preferred_brands": self.preferred_brands,
            "sizes": self.sizes,
            "colors": self.colors,
            "categories": self.categories
        }

# Demo: Extract and remember preferences
print("🎯 SELECTIVE MEMORY — User Preferences\n" + "="*60)

prefs = ShoppingPreferences()

# Simulate conversation and extract preferences
conversation = [
    ("I'm looking for laptops under $1200", "budget", 1200),
    ("I prefer Dell or Apple", "preferred_brands", ["Dell", "Apple"]),
    ("I like silver or space gray", "colors", ["silver", "space gray"]),
    ("I wear size 10 shoes", "sizes", {"shoes": 10}),
]

for msg, key, value in conversation:
    print(f"\n👤 Customer: {msg}")
    prefs.update(key, value)
    print(f"   🧠 Remembered: {key} = {value}")

print("\n" + "="*60)
print("🎯 USER PREFERENCE PROFILE:")
print(json.dumps(prefs.to_dict(), indent=2))

print("\n💡 Benefits:")
print("  ✅ Personalized recommendations")
print("  ✅ Auto-filter products by preferences")
print("  ✅ Reduce repetitive questions")
print("  ✅ Improve conversion rates")
```

## 🤔 Reflection Questions

**Q1: When should you use ConversationSummaryMemory vs ConversationBufferWindowMemory?**
<details>
<summary>Click for answer</summary>
<strong>Use ConversationSummaryMemory when:</strong>
<ul>
<li>Conversations are very long (50+ turns)</li>
<li>You need to remember the gist but not exact wording</li>
<li>Token costs are a concern</li>
<li>Examples: Therapy chatbots, long support sessions</li>
</ul>
<strong>Use ConversationBufferWindowMemory when:</strong>
<ul>
<li>Only recent context matters (e.g., form filling)</li>
<li>Exact wording is important</li>
<li>Conversations have natural "chapters" or topics</li>
<li>Examples: Multi-step forms, technical troubleshooting</li>
</ul>
<strong>Best practice:</strong> Start with ConversationSummaryBufferMemory (hybrid) for most production use cases.
</details>

**Q2: How do you handle memory for anonymous users who later log in?**
<details>
<summary>Click for answer</summary>
<strong>Pattern: Session Migration</strong>
<ol>
<li><strong>Anonymous:</strong> Store memory with session_id (e.g., UUID)</li>
<li><strong>Login:</strong> Migrate session_id memory to user_id</li>
<li><strong>Merge:</strong> If user has existing history, append new conversation</li>
</ol>
<br>
<strong>Example:</strong>
<pre>
# Anonymous user browses
session_123: "Looking for laptops" -> "Showed Dell XPS"

# User logs in as user_456
user_456: {previous history} + session_123 messages

# Continue with personalized context
</pre>
<strong>Implementation:</strong> Use database with session_id and user_id columns, migrate on auth.
</details>

**Q3: What are the privacy and security concerns with conversation memory?**
<details>
<summary>Click for answer</summary>
<strong>Key Concerns:</strong>
<ol>
<li><strong>PII Storage:</strong> Conversation may contain names, emails, credit cards</li>
<li><strong>Data Retention:</strong> How long to keep conversation history?</li>
<li><strong>Cross-User Leaks:</strong> Ensure user A cannot access user B's memory</li>
<li><strong>LLM Provider Access:</strong> Third-party LLMs see all memory</li>
</ol>
<strong>Best Practices:</strong>
<ul>
<li>✅ Encrypt memory at rest and in transit</li>
<li>✅ Implement data retention policies (delete after 30 days)</li>
<li>✅ Use user_id scoped memory (never global memory)</li>
<li>✅ Redact PII before storing ("John Smith" → "[NAME]")</li>
<li>✅ Allow users to delete conversation history (GDPR right to erasure)</li>
<li>✅ Use self-hosted LLMs for sensitive data</li>
</ul>
</details>

## 📝 Week 2 Project: Restaurant Reservation Chatbot

**Build a complete restaurant reservation system that:**

1. **Collects Information** (multi-turn):
   - Party size (2-10 people)
   - Date and time
   - Dietary restrictions
   - Special occasions (birthday, anniversary)

2. **Remembers Context:**
   - Previous visits and favorite dishes
   - Seating preferences (indoor/outdoor, booth/table)
   - Allergies

3. **Handles Modifications:**
   - "Actually, make it 6 people instead of 4"
   - "Change the time to 7pm"
   - "Add a high chair for a baby"

4. **Confirms and Summarizes:**
   - Repeat all details before confirming
   - Send confirmation with reservation ID
   - Provide cancellation instructions

**Required Features:**
- Use ConversationSummaryBufferMemory
- Track form completion progress
- Handle ambiguous inputs gracefully
- Multi-user session support

**Deliverables:**
- Working chatbot with memory
- Test with 3 different reservation scenarios
- Handle at least one modification per reservation
- Bonus: Add user preference storage ("Remember I'm vegetarian")

**Evaluation Criteria:**
- ✅ Collects all required information
- ✅ Remembers context across turns
- ✅ Handles modifications correctly
- ✅ Provides clear confirmation summary
- ✅ Code uses proper memory management

**Starter Code:**

```python
# Restaurant reservation chatbot starter

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Reservation(BaseModel):
    """Restaurant reservation details."""
    party_size: Optional[int] = Field(None, ge=1, le=10)
    date: Optional[str] = None  # YYYY-MM-DD
    time: Optional[str] = None  # HH:MM
    dietary_restrictions: list[str] = Field(default_factory=list)
    special_occasion: Optional[str] = None
    seating_preference: Optional[str] = None  # indoor/outdoor
    
    def is_complete(self):
        """Check if all required fields are filled."""
        return all([
            self.party_size is not None,
            self.date is not None,
            self.time is not None
        ])
    
    def to_summary(self):
        """Generate confirmation summary."""
        return f"""
        📋 RESERVATION SUMMARY
        Party Size: {self.party_size} guests
        Date: {self.date}
        Time: {self.time}
        Dietary Restrictions: {', '.join(self.dietary_restrictions) or 'None'}
        Special Occasion: {self.special_occasion or 'None'}
        Seating: {self.seating_preference or 'No preference'}
        """

# TODO: Build chatbot
# 1. Create memory (ConversationSummaryBufferMemory)
# 2. Create prompt that guides user through reservation
# 3. Build chain
# 4. Track reservation state
# 5. Handle modifications
# 6. Confirm when complete

# Test scenarios:
test_conversations = [
    # Scenario 1: Simple reservation
    [
        "I'd like to make a reservation",
        "For 4 people",
        "This Friday at 7pm",
        "No dietary restrictions",
        "Confirm"
    ],
    
    # Scenario 2: With modifications
    [
        "Table for 2 on Saturday at 6pm",
        "Actually, make it 3 people",
        "And change time to 7:30pm",
        "One person is vegetarian",
        "It's our anniversary",
        "Confirm"
    ],
    
    # Scenario 3: Complex with preferences
    [
        "Need a table for 6",
        "Next Tuesday at 8pm",
        "We need outdoor seating if possible",
        "Two people have nut allergies",
        "One is gluten-free",
        "It's a birthday party",
        "Confirm"
    ]
]

# Your implementation here!
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Memory fundamentals:**
- Why stateful conversations matter for user experience
- Four main memory types and their trade-offs
- Token usage vs context retention balance

✅ **Conversation patterns:**
- Multi-turn dialogues with context
- Session management for multiple users
- Persistent memory across sessions

✅ **Real-world application:**
- Built e-commerce shopping assistant
- Implemented preference tracking
- Handled cart management

✅ **Best practices:**
- Choose memory type based on use case
- Isolate user sessions properly
- Handle PII and security concerns
- Optimize for token efficiency

## 🔜 Next Week: Agents & Tools

In Week 3, we'll build **agents** that can use tools:
- ReAct (Reasoning + Acting) framework
- Tool creation (search, calculate, query)
- Agent executors and error handling
- Building autonomous research assistants

**Preview question:** How would you give the shopping assistant the ability to check real-time inventory and process payments?

## 📚 Additional Resources

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [Conversation Patterns Guide](https://python.langchain.com/docs/use_cases/chatbots/)
- [Token Optimization Strategies](https://platform.openai.com/docs/guides/prompt-engineering)

## 💡 Troubleshooting Tips

**Issue:** Bot forgets context after a few turns
- **Solution:** Check memory type - you might be using WindowMemory with k too small

**Issue:** Token limit exceeded errors
- **Solution:** Switch from BufferMemory to SummaryMemory or reduce max_token_limit

**Issue:** Responses are slow
- **Solution:** SummaryMemory makes extra LLM calls - use BufferMemory for short conversations

**Issue:** User A sees User B's conversation
- **Solution:** Ensure you're using user_id scoped memory, not global memory instance

---

**🎉 Congratulations on completing Week 2!** You can now build stateful conversational AI that provides personalized, context-aware experiences. See you next week! 🚀
