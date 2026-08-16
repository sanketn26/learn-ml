# Week 1 — LangChain Fundamentals & Basic Chains

**Course:** LangChain for AI Applications  
**Week Focus:** Understand prompt templates, structured output, and chain composition well enough to build and modify a small application.

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
- Understand the architecture of modern LLM applications
- Build reusable prompt templates with variable injection
- Compose simple and sequential chains using LCEL (LangChain Expression Language)
- Parse and validate LLM outputs with structured schemas
- Handle errors gracefully and implement retry logic
- Build a real-world customer support automation system

## 📊 Real-World Context

**The Challenge:** Your SaaS company handles 500+ support tickets daily:
- 40% are common questions (password resets, billing inquiries)
- 30% are bug reports requiring technical triage
- 20% are feature requests needing product team routing
- 10% are complex issues requiring human escalation

**The Solution:** An intelligent triage system that:
1. **Classifies** incoming tickets automatically (bug/feature/question/urgent)
2. **Generates** initial responses for common issues
3. **Extracts** key information (error messages, account IDs, reproduction steps)
4. **Routes** complex cases to appropriate teams
5. **Learns** from human feedback to improve over time

**Business Impact:**
- ⏱️ Reduce avg response time from 4 hours → 5 minutes
- 💰 Save $120K/year in support costs
- 😊 Increase customer satisfaction by 35%
- 📈 Scale support without linear headcount growth

Companies like **Intercom, Zendesk, and Notion** use similar LangChain-powered systems in production.


## 🔍 Part 1: The Problem — Why LangChain?

### Working Directly with LLM APIs (The Hard Way)

Let's see what building an AI application looks like *without* LangChain:

```python
# Example: Raw OpenAI API call (DON'T DO THIS IN PRODUCTION!)
# This is verbose, error-prone, and hard to maintain

import os
# Uncomment to use (requires OpenAI API key)
# import openai

# Problems with this approach:
raw_code_example = '''
# ❌ Problem 1: Hardcoded prompts (no reusability)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a support agent for CloudWave SaaS"},
        {"role": "user", "content": "My dashboard is loading slowly"}
    ],
    temperature=0.7,
)

# ❌ Problem 2: Deep nested JSON access
answer = response['choices'][0]['message']['content']

# ❌ Problem 3: No output validation
# What if the LLM returns invalid JSON? What if it hallucinates?

# ❌ Problem 4: No error handling
# What if API times out? Rate limits? Network errors?

# ❌ Problem 5: No composability
# How do you chain multiple LLM calls? Add memory? Use tools?
'''

print("🚨 Challenges with raw LLM APIs:")
print("1. Verbose boilerplate code")
print("2. No type safety or validation")
print("3. Hard to test and debug")
print("4. Difficult to chain operations")
print("5. No standardization across providers (OpenAI vs Anthropic vs Llama)")
print("\n💡 Solution: LangChain abstracts these complexities!")
```

### The LangChain way: cleaner composition

LangChain provides:
1. **Abstraction**: Unified interface for all LLM providers
2. **Composability**: Build complex workflows from simple components
3. **Type Safety**: Pydantic models for input/output validation
4. **Observability**: Built-in logging, tracing, and debugging
5. **Ecosystem**: 100+ integrations (databases, APIs, tools)

Let's see the same example, done properly:

```python
# Installation (uncomment to run):
# !pip install langchain langchain-openai python-dotenv

# For this demo, we'll use a fake LLM to avoid needing API keys
from langchain.llms.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ Clean, reusable code with LangChain

# 1. Create a prompt template (reusable across all tickets)
prompt = ChatPromptTemplate.from_template(
    """You are a helpful support agent for CloudWave SaaS.
    
Customer Issue: {user_issue}
Customer Tier: {tier}

Provide a helpful, professional response."""
)

# 2. Create LLM (swap providers easily: OpenAI, Anthropic, Llama, etc.)
# For demo purposes, using a fake LLM with predefined responses
llm = FakeListLLM(responses=[
    "Thank you for reporting this issue. Slow dashboard loading can be caused by browser cache. Try clearing your cache (Ctrl+Shift+Del) and reload. If the issue persists, please share your browser version and we'll investigate further."
])

# 3. Create output parser
parser = StrOutputParser()

# 4. Compose chain using LCEL (LangChain Expression Language)
chain = prompt | llm | parser

# 5. Invoke with inputs
result = chain.invoke({
    "user_issue": "My dashboard is loading slowly",
    "tier": "Enterprise"
})

print("🤖 AI Response:")
print(result)
print("\n✅ Benefits: Readable, testable, swappable components!")
```

## 📚 Part 2: Core Concepts Deep Dive

### 2.1 Prompt Templates — Consistency at Scale

**Why Templates Matter:**
- Reusability: Write once, use 10,000 times
- Consistency: Same quality across all interactions
- Version Control: Track prompt improvements
- A/B Testing: Compare prompt variations

**Real-World Example:** GitHub Copilot uses prompt templates to generate code consistently for millions of developers.

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# 🔹 Example 1: Simple String Template
simple_template = PromptTemplate.from_template(
    "Translate this to {language}: {text}"
)

print("📝 Simple Template:")
print(simple_template.format(language="Spanish", text="Hello, world!"))
print()

# 🔹 Example 2: Chat Template with System Message
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {role}. Be {tone}."),
    ("human", "{user_input}")
])

messages = chat_template.format_messages(
    role="Python developer",
    tone="concise and practical",
    user_input="How do I read a CSV file?"
)

print("💬 Chat Template:")
for msg in messages:
    print(f"{msg.type}: {msg.content}")
print()

# 🔹 Example 3: Few-Shot Learning Template
few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "Classify customer sentiment as positive, neutral, or negative."),
    ("human", "This product is amazing!"),
    ("assistant", "positive"),
    ("human", "It's okay, nothing special."),
    ("assistant", "neutral"),
    ("human", "Worst purchase ever!"),
    ("assistant", "negative"),
    ("human", "{new_review}")
])

print("🎯 Few-Shot Template (with examples):")
messages = few_shot_template.format_messages(
    new_review="The interface is intuitive and fast!"
)
print(f"User review: {messages[-1].content}")
print("✅ LLM will learn from examples and classify correctly")
```

### 2.2 Chains — Composing Complex Workflows

**LCEL (LangChain Expression Language)** uses the `|` pipe operator:

```python
chain = component1 | component2 | component3
```

Think of it like Unix pipes: `cat file.txt | grep "error" | wc -l`

**Benefits:**
- Readable: Left-to-right data flow
- Composable: Mix and match components
- Async-ready: Automatic parallelization
- Streaming: Real-time token generation

```python
from langchain.llms.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# 🔹 Example 1: Simple Linear Chain
# Flow: Prompt → LLM → String Output

prompt = ChatPromptTemplate.from_template(
    "Explain {concept} in one sentence for a {audience}."
)
llm = FakeListLLM(responses=[
    "Machine learning is teaching computers to learn from data instead of explicit programming."
])
parser = StrOutputParser()

simple_chain = prompt | llm | parser

result = simple_chain.invoke({
    "concept": "machine learning",
    "audience": "5-year-old"
})

print("🔗 Simple Chain Output:")
print(result)
print()

# 🔹 Example 2: Structured Output Chain
# Flow: Prompt → LLM → JSON Parser → Validated Object

class TicketAnalysis(BaseModel):
    """Structured analysis of a support ticket."""
    category: str = Field(description="bug, feature, question, or urgent")
    priority: int = Field(description="1-5, where 5 is highest")
    sentiment: str = Field(description="positive, neutral, or negative")
    key_points: list[str] = Field(description="main issues mentioned")
    suggested_response: str = Field(description="draft response")

structured_prompt = ChatPromptTemplate.from_template(
    """Analyze this support ticket:

Ticket: {ticket_text}

Provide analysis in JSON format:
{{"category": "...", "priority": ..., "sentiment": "...", "key_points": [...], "suggested_response": "..."}}
"""
)

structured_llm = FakeListLLM(responses=[
    '{"category": "bug", "priority": 4, "sentiment": "negative", "key_points": ["export button broken", "happens on Chrome"], "suggested_response": "Thank you for reporting this. We are investigating the export issue on Chrome browsers and will update you within 24 hours."}'
])

json_parser = JsonOutputParser(pydantic_object=TicketAnalysis)

structured_chain = structured_prompt | structured_llm | json_parser

analysis = structured_chain.invoke({
    "ticket_text": "The export button doesn't work! I've tried 5 times on Chrome. This is blocking my work."
})

print("📊 Structured Analysis:")
print(f"Category: {analysis['category']}")
print(f"Priority: {analysis['priority']}/5")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Key Points: {', '.join(analysis['key_points'])}")
print(f"Response: {analysis['suggested_response']}")
```

## 🛠️ Part 3: Building a Real Support Triage System

<div class="scenario-box">
<strong>📌 Scenario:</strong> You're building an intelligent ticket triage system that:
<ol>
<li>Accepts raw support tickets (email, chat, or web form)</li>
<li>Classifies tickets by type and urgency</li>
<li>Extracts key information (error codes, user IDs, steps to reproduce)</li>
<li>Generates draft responses for common issues</li>
<li>Routes complex cases to appropriate teams</li>
</ol>
</div>

### Step 1: Define the Data Model

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class TicketCategory(str, Enum):
    BUG = "bug"
    FEATURE = "feature_request"
    QUESTION = "question"
    BILLING = "billing"
    URGENT = "urgent"

class TicketPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class SupportTicketTriage(BaseModel):
    """Complete triage analysis of a support ticket."""
    category: TicketCategory = Field(description="Ticket classification")
    priority: TicketPriority = Field(description="Urgency level (1-5)")
    sentiment: str = Field(description="Customer sentiment: positive, neutral, frustrated, or angry")
    
    # Extracted Information
    error_code: Optional[str] = Field(default=None, description="Error code if mentioned")
    affected_feature: Optional[str] = Field(default=None, description="Product feature mentioned")
    user_id: Optional[str] = Field(default=None, description="User ID if mentioned")
    
    # Analysis
    key_issues: list[str] = Field(description="Main problems mentioned")
    reproduction_steps: Optional[list[str]] = Field(default=None, description="Steps to reproduce (if bug)")
    
    # Recommendations
    suggested_response: str = Field(description="Draft response to customer")
    assign_to_team: str = Field(description="engineering, product, billing, or support")
    escalate_to_human: bool = Field(description="Needs immediate human attention?")
    
    # SLA
    estimated_resolution_hours: int = Field(description="Expected time to resolve")

# Test the model
sample_triage = SupportTicketTriage(
    category=TicketCategory.BUG,
    priority=TicketPriority.HIGH,
    sentiment="frustrated",
    error_code="ERR_500",
    affected_feature="export",
    user_id="user_12345",
    key_issues=["export fails", "timeout error"],
    reproduction_steps=["Click export button", "Select CSV format", "Error appears"],
    suggested_response="We've identified the export timeout issue...",
    assign_to_team="engineering",
    escalate_to_human=True,
    estimated_resolution_hours=4
)

print("📋 Triage Data Model:")
print(sample_triage.model_dump_json(indent=2))
```

### Step 2: Build the Triage Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms.fake import FakeListLLM

# Create sophisticated prompt for ticket analysis
triage_prompt = ChatPromptTemplate.from_template(
    """You are an expert support ticket triage AI for CloudWave SaaS.

Analyze this support ticket and provide a comprehensive triage assessment.

TICKET INFORMATION:
From: {customer_email}
Subject: {subject}
Message:
{message}

CUSTOMER CONTEXT:
- Plan: {customer_plan}
- Account Age: {account_age_days} days
- Previous Tickets: {previous_ticket_count}

INSTRUCTIONS:
1. Classify the ticket (bug/feature_request/question/billing/urgent)
2. Assign priority (1=low to 5=emergency)
3. Assess customer sentiment
4. Extract: error codes, affected features, user IDs
5. List key issues and reproduction steps (if applicable)
6. Draft a professional, helpful response
7. Recommend team assignment (engineering/product/billing/support)
8. Determine if human escalation is needed
9. Estimate resolution time in hours

RESPONSE FORMAT:
{format_instructions}

Analyze now:"""
)

# Create parser with schema
parser = JsonOutputParser(pydantic_object=SupportTicketTriage)

# Add format instructions to prompt
triage_prompt = triage_prompt.partial(
    format_instructions=parser.get_format_instructions()
)

# Create LLM (using fake for demo - replace with real LLM)
llm = FakeListLLM(responses=[
    '''{
        "category": "bug",
        "priority": 4,
        "sentiment": "frustrated",
        "error_code": "ERR_TIMEOUT_500",
        "affected_feature": "data export",
        "user_id": "alice@example.com",
        "key_issues": ["export times out after 30 seconds", "affects large datasets", "blocking critical workflow"],
        "reproduction_steps": ["Navigate to Analytics dashboard", "Select 'Export to CSV'", "Choose date range > 6 months", "Click Export button", "Wait 30 seconds - timeout occurs"],
        "suggested_response": "Thank you for reporting this, Alice. I see you're experiencing timeout issues when exporting large datasets. This is a known issue affecting datasets over 100K rows. Our engineering team is implementing a fix with chunked exports. As a temporary workaround, try exporting smaller date ranges (1-3 months). We'll update you within 24 hours with the permanent fix timeline. We apologize for the disruption to your workflow.",
        "assign_to_team": "engineering",
        "escalate_to_human": true,
        "estimated_resolution_hours": 24
    }'''
])

# Build the triage chain
triage_chain = triage_prompt | llm | parser

print("✅ Triage chain created!")
print("Components: Prompt → LLM → JSON Parser → Validated Triage Object")
```

### Step 3: Test with Real-World Tickets

```python
# Simulate a real support ticket
ticket_input = {
    "customer_email": "alice@techcorp.com",
    "subject": "Urgent: Export feature broken for large datasets",
    "message": """Hi support team,
    
    I'm trying to export our Q4 analytics data (about 150K rows) and the export keeps timing out after 30 seconds with error ERR_TIMEOUT_500.
    
    Steps I've tried:
    1. Go to Analytics > Data Export
    2. Select date range: Oct 1 - Dec 31
    3. Choose CSV format
    4. Click Export
    5. Wait... then timeout error
    
    This is blocking our quarterly board presentation tomorrow. Please help ASAP!
    
    Thanks,
    Alice
    """,
    "customer_plan": "Enterprise",
    "account_age_days": 180,
    "previous_ticket_count": 2
}

# Run triage
triage_result = triage_chain.invoke(ticket_input)

print("🎫 TICKET TRIAGE ANALYSIS")
print("=" * 60)
print(f"📧 From: {ticket_input['customer_email']}")
print(f"📌 Subject: {ticket_input['subject']}")
print()
print("📊 CLASSIFICATION:")
print(f"  Category: {triage_result['category'].upper()}")
print(f"  Priority: {triage_result['priority']}/5 ({'🔴 HIGH' if triage_result['priority'] >= 4 else '🟡 MEDIUM'})")
print(f"  Sentiment: {triage_result['sentiment']}")
print()
print("🔍 EXTRACTED INFO:")
print(f"  Error Code: {triage_result['error_code']}")
print(f"  Feature: {triage_result['affected_feature']}")
print(f"  User: {triage_result['user_id']}")
print()
print("🐛 KEY ISSUES:")
for i, issue in enumerate(triage_result['key_issues'], 1):
    print(f"  {i}. {issue}")
print()
print("🔄 REPRODUCTION STEPS:")
for i, step in enumerate(triage_result['reproduction_steps'], 1):
    print(f"  {i}. {step}")
print()
print("💬 SUGGESTED RESPONSE:")
print(f"  {triage_result['suggested_response']}")
print()
print("📋 ROUTING:")
print(f"  Assign to: {triage_result['assign_to_team'].upper()} team")
print(f"  Escalate: {'🚨 YES - Human review needed' if triage_result['escalate_to_human'] else '✅ No'}")
print(f"  ETA: {triage_result['estimated_resolution_hours']} hours")
print("=" * 60)
```

## 🤔 Reflection Questions

**Q1: When should you use temperature=0 vs 0.7 vs 1.0?**
<details>
<summary>Click for answer</summary>
<ul>
<li><strong>temperature=0:</strong> Deterministic, best for classification, extraction, structured tasks (e.g., ticket triage)</li>
<li><strong>temperature=0.7:</strong> Balanced creativity/consistency, good for drafting responses, general Q&A</li>
<li><strong>temperature=1.0+:</strong> Creative, good for brainstorming, content generation, multiple variations</li>
</ul>
<strong>Rule of thumb:</strong> Start at 0.7, lower for precision tasks, raise for creative tasks.
</details>

**Q2: How do you prevent LLM hallucinations in production?**
<details>
<summary>Click for answer</summary>
<ol>
<li><strong>Structured Outputs:</strong> Use Pydantic models + JSON parsing (constrains format)</li>
<li><strong>Explicit Instructions:</strong> "Only use information provided. Say 'I don't know' if uncertain."</li>
<li><strong>Few-Shot Examples:</strong> Show correct behavior in prompts</li>
<li><strong>Retrieval:</strong> Ground responses in actual data (we'll cover RAG in Week 4)</li>
<li><strong>Human Review:</strong> For high-stakes decisions, always require human approval</li>
</ol>
</details>

**Q3: When should you use Agents instead of Chains?**
<details>
<summary>Click for answer</summary>
<strong>Use Chains when:</strong>
<ul>
<li>Workflow is predictable (always same steps)</li>
<li>You want full control over execution</li>
<li>Performance/cost is critical (chains are faster/cheaper)</li>
</ul>
<strong>Use Agents when:</strong>
<ul>
<li>LLM needs to decide which tools to use</li>
<li>Number of steps varies based on input</li>
<li>Problem requires reasoning and planning</li>
</ul>
Example: Ticket triage = Chain (fixed steps). Research task = Agent (dynamic tool use).
</details>

## 📝 Week 1 Project: Email Triage System

**Build a complete email triage system that:**

1. **Accepts emails** with subject + body
2. **Classifies** into categories:
   - Spam (promotional, unsolicited)
   - Urgent (requires immediate action)
   - Normal (standard priority)
   - Follow-up (needs response)
3. **Extracts** action items (deadlines, tasks, requests)
4. **Generates** brief summary (1-2 sentences)
5. **Suggests** response (if applicable)

**Deliverables:**
- Pydantic model for email analysis
- Triage chain with prompt template
- Test with 5 different emails (include spam, urgent, normal)
- Bonus: Add sentiment analysis

**Evaluation Criteria:**
- ✅ Correct classification (compare to ground truth)
- ✅ Action items extracted accurately
- ✅ Summary is concise and relevant
- ✅ Code is clean and uses LCEL properly

**Starter Code:**

```python
# Email triage project starter

from pydantic import BaseModel, Field
from typing import Optional

class EmailTriage(BaseModel):
    """Complete email analysis."""
    category: str = Field(description="spam, urgent, normal, or follow_up")
    priority: int = Field(description="1-5")
    # TODO: Add more fields (action_items, summary, sentiment, etc.)

# TODO: Create prompt template
# TODO: Create LLM and parser
# TODO: Build chain
# TODO: Test with sample emails

# Sample test emails:
test_emails = [
    {
        "from": "boss@company.com",
        "subject": "URGENT: Client demo tomorrow at 9am",
        "body": "We need the slides ready by tonight. Can you send them by 8pm?"
    },
    {
        "from": "marketing@spam.com",
        "subject": "🎉 50% OFF - Limited Time Offer!!!",
        "body": "Buy now! Amazing deals! Click here!"
    },
    # Add 3 more diverse examples
]

# Your implementation here!
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **LangChain fundamentals:**
- Why LangChain > raw LLM APIs (composability, type safety, ecosystem)
- LCEL pipe syntax: `prompt | llm | parser`
- Prompt templates for consistency and reusability

✅ **Structured outputs:**
- Pydantic models for type-safe LLM responses
- JsonOutputParser for automatic validation
- Handling complex nested data structures

✅ **Real-world application:**
- Built production-grade support ticket triage system
- Extracted structured information from unstructured text
- Implemented routing logic and escalation rules

✅ **Best practices:**
- Temperature settings for different use cases
- Preventing hallucinations with constraints
- When to use chains vs agents

## 🔜 Next Week: Memory & Conversation

In Week 2, we'll add **memory** to our chains:
- Conversation history tracking
- Context window management
- Multi-turn dialogues
- Building stateful chatbots

**Preview question:** How would you modify the ticket triage system to remember previous customer interactions?

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

**🎉 Congratulations on completing Week 1!** You're now equipped to build LLM-powered applications with LangChain. See you next week! 🚀
