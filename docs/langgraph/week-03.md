# Week 3 — Persistence & Replay

**Course:** LangGraph for Complex Workflows  
**Week Focus:** Save, resume, and replay workflow executions for reliability and debugging.

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
- Understand state persistence and checkpointing
- Implement workflow save/resume mechanisms
- Replay workflows for debugging
- Handle long-running workflows (hours/days)
- Audit and trace workflow execution
- Build fault-tolerant systems

## 📊 Real-World Context

**The Problem:**
- Workflow runs for 30 minutes, fails at step 27
- Restart from beginning = waste 27 minutes
- What if intermediate results were saved?

**Persistence Solutions:**
1. **Checkpointing:** Save state every step
2. **Resume:** Start from last checkpoint, not beginning
3. **Replay:** Re-run from any checkpoint for testing
4. **History:** See what happened at each step

**Business Impact:**
- ⏱️ Long-running workflows: 10 min → 1 min recovery
- 🐛 Debugging: Replay specific scenarios instantly
- 📊 Audit: Full execution trail for compliance
- 🔄 Resilience: Survive infrastructure failures
- 💰 Save millions in compute costs (no repeated work)


## 🔍 Part 1: Understanding Checkpointing

<div class="checkpoint-box">
<strong>Checkpoint:</strong> A saved snapshot of workflow state at a specific point.
</div>

### Without Checkpointing (Naive)

```
Node A (30s) → Node B (20s) → Node C (25s) → Node D (fail)
│              │              │              │
done           done           done           ERROR!
│              │              │              │
Restart ───────────────────────────────────→ 
75 seconds wasted!
```

### With Checkpointing (Smart)

```
Node A (30s) → ✓ Checkpoint A
Node B (20s) → ✓ Checkpoint B  
Node C (25s) → ✓ Checkpoint C
Node D (fail) → ✗ ERROR

Resume from Checkpoint C → 
Node D (retry) → Success!

Only 25 seconds of recomputation!
```

## 📚 Part 2: Implementing Persistence

### Checkpoint Storage Options

| Store | Cost | Speed | Scale | Use Case |
|-------|------|-------|-------|----------|
| In-Memory | Free | Fastest | 1 machine | Development |
| Database | Low | Fast | Any | Production |
| Redis | Low | Very fast | Millions | High-throughput |
| S3/GCS | Very low | Slower | Unlimited | Long-term archive |

**Best Practice:** Use database for production (durable, queryable, scalable)

```python
# Implement a simple checkpoint system

from datetime import datetime
from typing import Any, Dict, Optional
import json
import uuid

class Checkpoint:
    """Represents a saved workflow state."""
    
    def __init__(self, node_name: str, state: Dict[str, Any]):
        self.id = str(uuid.uuid4())[:8]
        self.node_name = node_name
        self.state = state.copy()
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"Checkpoint({self.node_name}, {self.timestamp.strftime('%H:%M:%S')})"

class PersistenceStore:
    """Store checkpoints in memory (demo) or database (production)."""
    
    def __init__(self):
        self.checkpoints: Dict[str, list[Checkpoint]] = {}
    
    def save_checkpoint(self, workflow_id: str, checkpoint: Checkpoint) -> str:
        """Save a checkpoint."""
        if workflow_id not in self.checkpoints:
            self.checkpoints[workflow_id] = []
        
        self.checkpoints[workflow_id].append(checkpoint)
        return checkpoint.id
    
    def get_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if workflow_id in self.checkpoints and self.checkpoints[workflow_id]:
            return self.checkpoints[workflow_id][-1]
        return None
    
    def get_checkpoint_at_node(self, workflow_id: str, node_name: str) -> Optional[Checkpoint]:
        """Get checkpoint for a specific node."""
        if workflow_id in self.checkpoints:
            for cp in reversed(self.checkpoints[workflow_id]):
                if cp.node_name == node_name:
                    return cp
        return None
    
    def list_checkpoints(self, workflow_id: str):
        """List all checkpoints for a workflow."""
        return self.checkpoints.get(workflow_id, [])

# Demo: Creating checkpoints
store = PersistenceStore()
workflow_id = "report-2024-001"

print("💾 Creating Checkpoints")
print("="*70)

# Simulate workflow progression
state = {"data": [], "status": "starting"}

# Node 1: Extract Data
print("\n→ Node 1: Extracting data...")
state["data"] = ["record1", "record2", "record3"]
state["status"] = "extracted"
cp1 = Checkpoint("extract_data", state)
store.save_checkpoint(workflow_id, cp1)
print(f"  ✓ Saved checkpoint: {cp1}")

# Node 2: Clean Data
print("\n→ Node 2: Cleaning data...")
state["data"] = [r.upper() for r in state["data"]]
state["status"] = "cleaned"
cp2 = Checkpoint("clean_data", state)
store.save_checkpoint(workflow_id, cp2)
print(f"  ✓ Saved checkpoint: {cp2}")

# Node 3: Analyze Data
print("\n→ Node 3: Analyzing data...")
state["statistics"] = {"count": len(state["data"]), "sample": state["data"][0]}
state["status"] = "analyzed"
cp3 = Checkpoint("analyze_data", state)
store.save_checkpoint(workflow_id, cp3)
print(f"  ✓ Saved checkpoint: {cp3}")

print("\n" + "="*70)
print("\n📊 Checkpoint History:")
for i, cp in enumerate(store.list_checkpoints(workflow_id), 1):
    print(f"  {i}. {cp.node_name:20} | State keys: {list(cp.state.keys())}")

print(f"\nLatest checkpoint: {store.get_latest_checkpoint(workflow_id)}")
```

## 🔄 Part 3: Replay & Debugging

<div class="replay-box">
<strong>Replay:</strong> Re-run workflow from a saved checkpoint (with possibly different logic).
</div>

### Use Cases for Replay

**1. Debugging:** "Why did Node D fail?"
```
Original run: A → B → C → D ✗

Debug run: Resume from C, trace D execution step-by-step
```

**2. Testing Changes:** "Will the fix work?"
```
Original run: A → B → C → D ✗
Replay: Resume from C with fixed Node D code
```

**3. What-If Analysis:** "What if we changed Node B?"
```
Original run: A → B (variant 1) → C → D
What-if run: Resume from A, use B (variant 2) → C → D
```

```python
# Demonstrate replay capability

print("\n🔄 Replay: Resume from Checkpoint")
print("="*70)

# Get checkpoint to resume from
resume_checkpoint = store.get_checkpoint_at_node(workflow_id, "clean_data")

if resume_checkpoint:
    print(f"\n✓ Resuming from checkpoint: {resume_checkpoint.node_name}")
    print(f"  Timestamp: {resume_checkpoint.timestamp.strftime('%H:%M:%S')}")
    print(f"  State: {resume_checkpoint.state}")
    
    # Continue from this state
    state = resume_checkpoint.state.copy()
    
    print(f"\n→ Continuing from here...")
    
    # Skip Node 3 (already done), go to Node 4
    print(f"  Node 3: Skipped (already completed)")
    
    # Node 4: Generate Report (NEW)
    print(f"  Node 4: Generating report...")
    state["report"] = f"Analysis of {len(state['data'])} records"
    state["status"] = "completed"
    cp_final = Checkpoint("generate_report", state)
    store.save_checkpoint(workflow_id, cp_final)
    print(f"  ✓ Saved checkpoint: {cp_final}")
    
    print(f"\n✅ Workflow completed!")
    print(f"   Final state: {state}")
```
