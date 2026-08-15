# Week 1 — LangGraph Fundamentals & State Management

**Course:** LangGraph for Complex Workflows  
**Week Focus:** Master state graphs, conditional routing, and multi-step workflows to build production-grade AI systems.

---

## If you already write software

LangGraph is a **state machine**. Nodes are functions. Edges are control flow. State is the request-scoped object you thread through.

You have written this as:

- a workflow engine
- a Redux store + reducers
- a CI pipeline with conditional jobs
- an XState chart
- a saga

```
graph = StateGraph(State)
graph.add_node("parse", parse)       # a function: State -> partial State
graph.add_node("act", act)
graph.add_edge("parse", "act")       # always
graph.add_conditional_edges("act", route)   # if / else
```

The payoff versus a pile of `if` statements: you can **checkpoint**, **replay**, and **pause for a human** because the runtime owns the state. That is the point of the next three weeks. If your flow is three sequential LLM calls with no branch, a chain is enough — do not pay for a graph yet.

## 🎯 Learning Objectives

By the end of this week, you will:
- Understand why graphs are superior to linear chains for complex workflows
- Design and implement StateGraph with typed state schemas
- Build nodes (workflow steps) and edges (transitions)
- Implement conditional routing based on state
- Handle errors gracefully in graph execution
- Visualize and debug graph workflows
- Build a real-world document processing pipeline

## 📊 Real-World Context

**The Challenge:** Your content platform receives 10,000+ user submissions daily:
- 📄 Blog posts, comments, product reviews
- 🎭 Mix of legitimate content and spam/toxic material
- 🌍 Multiple languages requiring classification
- ⚖️ Need to moderate without human bottleneck

**Linear Chain Limitations:**
```python
# ❌ This doesn't work well:
chain = classify | moderate | summarize | publish
# Problem: What if we need to:
# - Route spam to deletion (skip summarization)
# - Send toxic content to human review
# - Handle multiple languages differently
# - Retry failed steps
```

**The Solution:** A content moderation graph that:
1. **Classifies** content type (article/comment/review/spam)
2. **Detects** language and toxicity
3. **Routes** based on results:
   - Spam → Auto-reject
   - Toxic → Human review queue
   - Clean → Extract key info
4. **Summarizes** approved content
5. **Publishes** or routes for approval

**Business Impact:**
- 🚀 Process 10K submissions/day (up from 500 manual reviews)
- ⏱️ Reduce moderation time from 4 hours → 2 minutes
- 🎯 95% accuracy with 10% human review (high-risk items)
- 💰 Save $240K/year in moderation costs

Companies like **Reddit, Medium, and Substack** use similar graph-based moderation systems.


## 🔍 Part 1: Why Graphs? (Linear Chains vs State Graphs)

### The Problem with Linear Chains

LangChain chains work great for simple workflows, but fail for complex scenarios:

```python
# ❌ Linear Chain - Can't handle branching logic

from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser

print("❌ LINEAR CHAIN LIMITATIONS:")
print()

# Example: Content moderation
prompt = ChatPromptTemplate.from_template("Moderate this content: {content}")
llm = FakeListLLM(responses=["Content is spam"])
parser = StrOutputParser()

chain = prompt | llm | parser

print("Problem 1: No conditional routing")
print("  If content is spam, we should STOP here.")
print("  But chains always run ALL steps.")
print()

print("Problem 2: No shared state")
print("  Each step only sees the previous output.")
print("  Can't access original input or intermediate results.")
print()

print("Problem 3: No cycles/loops")
print("  Can't retry failed steps.")
print("  Can't implement 'try again until success' logic.")
print()

print("Problem 4: Hard to debug")
print("  Can't inspect state between steps.")
print("  Can't visualize the workflow.")
print()

print("💡 Solution: Use LangGraph!")
```

### The LangGraph Way (State Graphs)

LangGraph introduces:
1. **State**: Shared context accessible by all nodes
2. **Nodes**: Functions that read/write state
3. **Conditional Edges**: Dynamic routing based on state
4. **Cycles**: Loops and retries
5. **Visualization**: See your workflow as a graph

```python
# ✅ State Graph - Handles complexity elegantly

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# 1. Define State Schema
class ContentState(TypedDict):
    """Shared state across all nodes."""
    content: str
    content_type: str  # "spam", "toxic", "clean"
    decision: str       # "reject", "review", "approve"
    summary: str

# 2. Define Node Functions
def classify_content(state: ContentState) -> ContentState:
    """Node 1: Classify content."""
    # In real app: use LLM to classify
    if "buy now" in state["content"].lower():
        state["content_type"] = "spam"
    elif "hate" in state["content"].lower():
        state["content_type"] = "toxic"
    else:
        state["content_type"] = "clean"
    return state

def route_decision(state: ContentState) -> Literal["reject", "review", "approve"]:
    """Conditional router: decide next step based on state."""
    if state["content_type"] == "spam":
        return "reject"
    elif state["content_type"] == "toxic":
        return "review"
    else:
        return "approve"

def reject_content(state: ContentState) -> ContentState:
    """Node 2a: Auto-reject spam."""
    state["decision"] = "rejected"
    state["summary"] = "Spam detected - auto-rejected"
    return state

def queue_for_review(state: ContentState) -> ContentState:
    """Node 2b: Queue for human review."""
    state["decision"] = "needs_review"
    state["summary"] = "Toxic content - queued for human review"
    return state

def approve_content(state: ContentState) -> ContentState:
    """Node 2c: Auto-approve clean content."""
    state["decision"] = "approved"
    state["summary"] = "Clean content - approved for publication"
    return state

# 3. Build the Graph
workflow = StateGraph(ContentState)

# Add nodes
workflow.add_node("classify", classify_content)
workflow.add_node("reject", reject_content)
workflow.add_node("review", queue_for_review)
workflow.add_node("approve", approve_content)

# Set entry point
workflow.set_entry_point("classify")

# Add conditional edges (routing logic)
workflow.add_conditional_edges(
    "classify",
    route_decision,
    {
        "reject": "reject",
        "review": "review",
        "approve": "approve"
    }
)

# All paths end after their respective actions
workflow.add_edge("reject", END)
workflow.add_edge("review", END)
workflow.add_edge("approve", END)

# 4. Compile the graph
app = workflow.compile()

print("✅ Graph created successfully!")
print("\n📊 GRAPH STRUCTURE:")
print("""
    START
      |
      v
  [classify]
      |
   <router>
   /  |  \\
  /   |   \\
spam toxic clean
 |    |     |
 v    v     v
[reject] [review] [approve]
 |    |     |
 v    v     v
    END
""")
```

### Test the Graph with Different Inputs

```python
# Test 1: Spam content
print("🧪 TEST 1: Spam Content")
print("=" * 60)
result1 = app.invoke({"content": "BUY NOW! Limited time offer!!!"})
print(f"Input: {result1['content']}")
print(f"Type: {result1['content_type']}")
print(f"Decision: {result1['decision']}")
print(f"Summary: {result1['summary']}")
print()

# Test 2: Toxic content
print("🧪 TEST 2: Toxic Content")
print("=" * 60)
result2 = app.invoke({"content": "I hate this product and everyone who uses it!"})
print(f"Input: {result2['content']}")
print(f"Type: {result2['content_type']}")
print(f"Decision: {result2['decision']}")
print(f"Summary: {result2['summary']}")
print()

# Test 3: Clean content
print("🧪 TEST 3: Clean Content")
print("=" * 60)
result3 = app.invoke({"content": "This is a helpful tutorial on Python programming."})
print(f"Input: {result3['content']}")
print(f"Type: {result3['content_type']}")
print(f"Decision: {result3['decision']}")
print(f"Summary: {result3['summary']}")
print()

print("✅ Notice how each input takes a DIFFERENT path through the graph!")
```

## 📚 Part 2: Core Concepts Deep Dive

### 2.1 State Schemas — The Heart of LangGraph

**State** is a shared dictionary that flows through the graph. Every node can:
- **Read** from state
- **Write** to state (updates are merged)
- **Access** full history

**Best Practices:**
1. Use TypedDict for type safety
2. Document each field
3. Keep state flat (avoid deep nesting)
4. Use Optional for fields set later

```python
from typing import TypedDict, Optional, List
from datetime import datetime

# ✅ Good State Design
class DocumentProcessingState(TypedDict):
    """State for multi-step document processing workflow."""
    
    # Input (set at start)
    document_text: str
    document_id: str
    
    # Classification results (set by classify node)
    document_type: Optional[str]  # "invoice", "contract", "report"
    language: Optional[str]        # "en", "es", "fr"
    confidence: Optional[float]    # 0.0-1.0
    
    # Extraction results (set by extract node)
    entities: Optional[List[dict]]  # [{"type": "person", "value": "John"}]
    key_dates: Optional[List[str]]  # ["2024-01-15", "2024-02-01"]
    amounts: Optional[List[float]]  # [1500.00, 2300.50]
    
    # Summary (set by summarize node)
    summary: Optional[str]
    
    # Routing decision (set by router)
    next_step: Optional[str]  # "approve", "reject", "review"
    
    # Metadata
    processed_at: Optional[str]
    errors: Optional[List[str]]

print("✅ Well-designed state schema!")
print("\nKey features:")
print("1. Clear input vs output fields")
print("2. Optional fields for values set later")
print("3. Specific types (List[dict], float, etc.)")
print("4. Docstrings for clarity")
print("5. Error tracking built-in")
```

### 2.2 Nodes — The Workflow Steps

**Nodes** are functions that:
- Take state as input
- Perform work (call LLM, API, database, etc.)
- Return updated state

**Node Types:**
1. **Processing nodes**: Transform data (classify, extract, summarize)
2. **Decision nodes**: Analyze state and set routing flags
3. **Integration nodes**: Call external APIs/databases
4. **Validation nodes**: Check data quality

```python
from typing import TypedDict
from datetime import datetime

class DocState(TypedDict):
    text: str
    doc_type: str
    entities: list
    summary: str
    error: str
    timestamp: str

# Example 1: Processing Node
def classify_document(state: DocState) -> DocState:
    """Classify document type using keyword matching."""
    text_lower = state["text"].lower()
    
    if "invoice" in text_lower or "payment" in text_lower:
        state["doc_type"] = "invoice"
    elif "agreement" in text_lower or "contract" in text_lower:
        state["doc_type"] = "contract"
    else:
        state["doc_type"] = "report"
    
    state["timestamp"] = datetime.now().isoformat()
    return state

# Example 2: Extraction Node
def extract_entities(state: DocState) -> DocState:
    """Extract key entities from document."""
    # In production: use NER model or LLM
    entities = []
    
    # Simple extraction example
    if "$" in state["text"]:
        entities.append({"type": "amount", "value": "$1,500"})
    
    state["entities"] = entities
    return state

# Example 3: Summarization Node (with LLM)
def summarize_document(state: DocState) -> DocState:
    """Generate concise summary."""
    # In production: use actual LLM
    doc_type = state.get("doc_type", "document")
    state["summary"] = f"This is a {doc_type} containing {len(state['text'])} characters."
    return state

# Example 4: Error Handling Node
def validate_document(state: DocState) -> DocState:
    """Validate document before processing."""
    if not state.get("text"):
        state["error"] = "Empty document"
    elif len(state["text"]) < 10:
        state["error"] = "Document too short"
    else:
        state["error"] = ""  # No error
    return state

print("✅ Node functions created!")
print("\n💡 Node Best Practices:")
print("1. Single responsibility (do ONE thing well)")
print("2. Always return state (even if unchanged)")
print("3. Handle errors gracefully (don't crash)")
print("4. Add logging for debugging")
print("5. Keep nodes pure (no hidden side effects)")
```

### 2.3 Edges — Connecting the Workflow

**Edge Types:**

1. **Normal Edges**: Always go from A → B
   ```python
   workflow.add_edge("node_a", "node_b")
   ```

2. **Conditional Edges**: Route based on state
   ```python
   workflow.add_conditional_edges(
       "router_node",
       routing_function,
       {"option1": "node_a", "option2": "node_b"}
   )
   ```

3. **Entry Point**: Where execution starts
   ```python
   workflow.set_entry_point("first_node")
   ```

4. **End**: Terminal node (no outgoing edges)
   ```python
   workflow.add_edge("final_node", END)
   ```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class SimpleState(TypedDict):
    value: int
    path_taken: str

# Example: Conditional routing based on value
def check_value(state: SimpleState) -> Literal["low", "high"]:
    """Router: decide path based on value."""
    return "low" if state["value"] < 50 else "high"

def process_low(state: SimpleState) -> SimpleState:
    state["path_taken"] = "LOW path"
    return state

def process_high(state: SimpleState) -> SimpleState:
    state["path_taken"] = "HIGH path"
    return state

# Build graph with conditional routing
graph = StateGraph(SimpleState)
graph.add_node("process_low", process_low)
graph.add_node("process_high", process_high)

graph.set_conditional_entry_point(
    check_value,
    {"low": "process_low", "high": "process_high"}
)

graph.add_edge("process_low", END)
graph.add_edge("process_high", END)

app = graph.compile()

# Test routing
print("🧪 Test conditional routing:")
print()
result1 = app.invoke({"value": 25})
print(f"Value=25 → {result1['path_taken']}")

result2 = app.invoke({"value": 75})
print(f"Value=75 → {result2['path_taken']}")

print("\n✅ Routing works! Different inputs take different paths.")
```

## 🛠️ Part 3: Building a Real Document Processing Pipeline

<div class="scenario-box">
<strong>📌 Scenario:</strong> Build an intelligent document processor for a financial services company:
<ol>
<li><strong>Classify</strong> document type (invoice, contract, report, form)</li>
<li><strong>Extract</strong> key information (dates, amounts, parties)</li>
<li><strong>Validate</strong> extracted data</li>
<li><strong>Summarize</strong> document content</li>
<li><strong>Route</strong> for appropriate action:
  <ul>
    <li>Invoice → Accounting system</li>
    <li>Contract → Legal review</li>
    <li>Report → Management dashboard</li>
    <li>Unknown → Human review</li>
  </ul>
</li>
</ol>
</div>

### Step 1: Define Comprehensive State

```python
from typing import TypedDict, Optional, List, Dict
from datetime import datetime

class DocumentState(TypedDict):
    """State for document processing workflow."""
    
    # Input
    document_id: str
    document_text: str
    source: str  # "email", "upload", "scan"
    
    # Classification
    document_type: Optional[str]  # "invoice", "contract", "report", "form", "unknown"
    classification_confidence: Optional[float]
    language: Optional[str]
    
    # Extraction
    entities: Optional[List[Dict[str, str]]]  # [{"type": "amount", "value": "$1500"}]
    dates: Optional[List[str]]
    amounts: Optional[List[float]]
    parties: Optional[List[str]]  # People/companies mentioned
    
    # Validation
    is_valid: Optional[bool]
    validation_errors: Optional[List[str]]
    
    # Summary
    summary: Optional[str]
    key_points: Optional[List[str]]
    
    # Routing
    routing_decision: Optional[str]  # "accounting", "legal", "management", "review"
    priority: Optional[str]  # "low", "medium", "high", "urgent"
    
    # Metadata
    processing_started: Optional[str]
    processing_completed: Optional[str]
    errors: Optional[List[str]]

print("✅ DocumentState schema defined")
print(f"\nTotal fields: {len(DocumentState.__annotations__)}")
print("Input fields: 3")
print("Processing fields: 14")
print("Metadata fields: 3")
```

### Step 2: Implement Processing Nodes

```python
import re
from datetime import datetime
from langchain.llms.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Node 1: Classify Document
def classify_document(state: DocumentState) -> DocumentState:
    """Classify document type using keyword analysis."""
    text_lower = state["document_text"].lower()
    
    # Classification logic (in production: use LLM)
    if any(word in text_lower for word in ["invoice", "payment", "bill", "amount due"]):
        state["document_type"] = "invoice"
        state["classification_confidence"] = 0.92
    elif any(word in text_lower for word in ["agreement", "contract", "hereby agree"]):
        state["document_type"] = "contract"
        state["classification_confidence"] = 0.88
    elif any(word in text_lower for word in ["report", "analysis", "findings"]):
        state["document_type"] = "report"
        state["classification_confidence"] = 0.85
    else:
        state["document_type"] = "unknown"
        state["classification_confidence"] = 0.40
    
    # Detect language (simplified)
    state["language"] = "en"  # Default to English
    
    state["processing_started"] = datetime.now().isoformat()
    return state

# Node 2: Extract Information
def extract_information(state: DocumentState) -> DocumentState:
    """Extract key entities, dates, amounts from document."""
    text = state["document_text"]
    
    # Extract dates (simple regex)
    date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
    dates = re.findall(date_pattern, text)
    state["dates"] = dates if dates else []
    
    # Extract amounts (simple regex)
    amount_pattern = r'\$\s*([0-9,]+\.?[0-9]*)'
    amounts_str = re.findall(amount_pattern, text)
    state["amounts"] = [float(amt.replace(',', '')) for amt in amounts_str]
    
    # Extract entities
    entities = []
    if state["amounts"]:
        entities.append({"type": "monetary_amount", "value": f"${state['amounts'][0]}"})
    if state["dates"]:
        entities.append({"type": "date", "value": state['dates'][0]})
    state["entities"] = entities
    
    # Extract parties (simplified - just capitalized words)
    parties = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
    state["parties"] = list(set(parties))[:5]  # Top 5 unique
    
    return state

# Node 3: Validate Data
def validate_extraction(state: DocumentState) -> DocumentState:
    """Validate extracted information."""
    errors = []
    
    # Validation rules
    if state["document_type"] == "invoice":
        if not state.get("amounts"):
            errors.append("Invoice missing amount")
        if not state.get("dates"):
            errors.append("Invoice missing date")
    
    if state["document_type"] == "contract":
        if not state.get("parties") or len(state.get("parties", [])) < 2:
            errors.append("Contract missing parties")
    
    if state["classification_confidence"] < 0.7:
        errors.append("Low classification confidence")
    
    state["is_valid"] = len(errors) == 0
    state["validation_errors"] = errors
    
    return state

# Node 4: Summarize Document
def summarize_document(state: DocumentState) -> DocumentState:
    """Generate summary and key points."""
    doc_type = state["document_type"]
    
    # Generate summary based on type
    if doc_type == "invoice":
        amount = state["amounts"][0] if state.get("amounts") else "unknown"
        date = state["dates"][0] if state.get("dates") else "unknown"
        state["summary"] = f"Invoice for ${amount} dated {date}"
        state["key_points"] = [
            f"Amount due: ${amount}",
            f"Date: {date}",
            f"Entities extracted: {len(state.get('entities', []))}"
        ]
    elif doc_type == "contract":
        parties = state.get("parties", [])
        state["summary"] = f"Contract agreement between {len(parties)} parties"
        state["key_points"] = [
            f"Parties: {', '.join(parties[:3])}",
            f"Dates mentioned: {len(state.get('dates', []))}"
        ]
    else:
        state["summary"] = f"{doc_type.title()} document with {len(state['document_text'])} characters"
        state["key_points"] = [f"Type: {doc_type}", f"Language: {state.get('language', 'unknown')}"]
    
    return state

# Node 5: Route Document
def route_document(state: DocumentState) -> DocumentState:
    """Determine routing and priority."""
    doc_type = state["document_type"]
    is_valid = state.get("is_valid", False)
    
    # Routing logic
    if not is_valid:
        state["routing_decision"] = "review"
        state["priority"] = "high"
    elif doc_type == "invoice":
        state["routing_decision"] = "accounting"
        # Check if urgent (amount > $10,000)
        amounts = state.get("amounts", [])
        state["priority"] = "urgent" if (amounts and amounts[0] > 10000) else "medium"
    elif doc_type == "contract":
        state["routing_decision"] = "legal"
        state["priority"] = "high"
    elif doc_type == "report":
        state["routing_decision"] = "management"
        state["priority"] = "low"
    else:
        state["routing_decision"] = "review"
        state["priority"] = "medium"
    
    state["processing_completed"] = datetime.now().isoformat()
    return state

print("✅ All node functions implemented!")
print("\nNodes created:")
print("  1. classify_document")
print("  2. extract_information")
print("  3. validate_extraction")
print("  4. summarize_document")
print("  5. route_document")
```

### Step 3: Build the Complete Graph

```python
from langgraph.graph import StateGraph, END

# Create the graph
workflow = StateGraph(DocumentState)

# Add all nodes
workflow.add_node("classify", classify_document)
workflow.add_node("extract", extract_information)
workflow.add_node("validate", validate_extraction)
workflow.add_node("summarize", summarize_document)
workflow.add_node("route", route_document)

# Define the workflow
workflow.set_entry_point("classify")
workflow.add_edge("classify", "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", "summarize")
workflow.add_edge("summarize", "route")
workflow.add_edge("route", END)

# Compile the graph
doc_processor = workflow.compile()

print("✅ Document processing graph compiled!")
print("\n📊 WORKFLOW:")
print("""
    START
      |
      v
  [classify] ← Determine document type
      |
      v
  [extract] ← Extract dates, amounts, entities
      |
      v
  [validate] ← Check data quality
      |
      v
  [summarize] ← Generate summary
      |
      v
  [route] ← Determine destination & priority
      |
      v
    END
""")
```

### Step 4: Test with Real-World Documents

```python
# Test Document 1: Invoice
invoice_doc = {
    "document_id": "DOC-001",
    "source": "email",
    "document_text": """
    INVOICE #INV-2024-001
    
    Date: 01/15/2024
    Due Date: 02/15/2024
    
    Bill To: Acme Corporation
    From: Tech Solutions Inc
    
    Services Rendered:
    - Software Development: $15,000.00
    - Cloud Hosting (Jan): $2,500.00
    
    Total Amount Due: $17,500.00
    
    Payment Terms: Net 30
    """
}

print("🧪 TEST 1: Processing Invoice")
print("=" * 70)
result = doc_processor.invoke(invoice_doc)

print(f"Document ID: {result['document_id']}")
print(f"Source: {result['source']}")
print()
print(f"📋 Classification:")
print(f"  Type: {result['document_type']}")
print(f"  Confidence: {result['classification_confidence']:.0%}")
print(f"  Language: {result['language']}")
print()
print(f"📊 Extraction:")
print(f"  Dates found: {len(result['dates'])} → {result['dates']}")
print(f"  Amounts found: {len(result['amounts'])} → ${result['amounts']}")
print(f"  Entities: {len(result['entities'])}")
for entity in result['entities']:
    print(f"    - {entity['type']}: {entity['value']}")
print()
print(f"✓ Validation:")
print(f"  Valid: {result['is_valid']}")
if result['validation_errors']:
    print(f"  Errors: {result['validation_errors']}")
print()
print(f"📝 Summary:")
print(f"  {result['summary']}")
print(f"  Key Points:")
for point in result['key_points']:
    print(f"    • {point}")
print()
print(f"🎯 Routing:")
print(f"  Destination: {result['routing_decision'].upper()}")
print(f"  Priority: {result['priority'].upper()}")
print()
print(f"⏱️ Processing Time: {result['processing_started'][:19]} → {result['processing_completed'][:19]}")
print("=" * 70)
print()
```

```python
# Test Document 2: Contract
contract_doc = {
    "document_id": "DOC-002",
    "source": "upload",
    "document_text": """
    SERVICE AGREEMENT
    
    This agreement is entered into on 2024-01-10 between:
    
    Party A: John Smith, representing Smith Enterprises LLC
    Party B: Jane Doe, representing Doe Consulting Inc
    
    The parties hereby agree to the following terms:
    
    1. Services: Consulting services for digital transformation
    2. Duration: 6 months starting 2024-02-01
    3. Compensation: $5,000.00 per month
    4. Termination: Either party may terminate with 30 days notice
    
    Signed on 2024-01-10
    """
}

print("🧪 TEST 2: Processing Contract")
print("=" * 70)
result = doc_processor.invoke(contract_doc)

print(f"📋 Classification: {result['document_type']} ({result['classification_confidence']:.0%})")
print(f"📊 Parties Identified: {', '.join(result['parties'])}")
print(f"✓ Valid: {result['is_valid']}")
print(f"📝 Summary: {result['summary']}")
print(f"🎯 Route to: {result['routing_decision'].upper()} (Priority: {result['priority'].upper()})")
print("=" * 70)
print()
```

```python
# Test Document 3: Unknown/Ambiguous
unknown_doc = {
    "document_id": "DOC-003",
    "source": "scan",
    "document_text": "Hello, this is a short note. Thanks!"
}

print("🧪 TEST 3: Processing Unknown Document")
print("=" * 70)
result = doc_processor.invoke(unknown_doc)

print(f"📋 Classification: {result['document_type']} ({result['classification_confidence']:.0%})")
print(f"✓ Valid: {result['is_valid']}")
if result['validation_errors']:
    print(f"⚠️ Validation Errors:")
    for error in result['validation_errors']:
        print(f"  • {error}")
print(f"🎯 Route to: {result['routing_decision'].upper()} (Priority: {result['priority'].upper()})")
print("\n✅ Notice: Low confidence → routed to human review!")
print("=" * 70)
```
