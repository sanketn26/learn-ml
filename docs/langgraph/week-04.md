# Week 4 — Human-in-the-Loop & Production

**Course:** LangGraph for Complex Workflows  
**Week Focus:** Pause workflows for human decisions and deploy to production.

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
- Pause workflows for human decisions
- Implement approval gates and feedback loops
- Route workflow based on human input
- Deploy interactive workflows at scale
- Handle SLA/timeout requirements
- Monitor human-in-the-loop metrics

## 📊 Real-World Context

**The Problem:**
- Fully automated workflows sometimes make wrong decisions
- High-stakes decisions need human review
- No good way to integrate humans into LLM workflows

**Human-in-the-Loop Solutions:**
1. **Approval Gates:** Pause before critical actions
2. **Feedback Loops:** Humans provide guidance
3. **Conditional Routing:** Different paths based on human decision
4. **Priority Escalation:** Complex cases go to senior reviewers

**Business Impact:**
- 🛡️ Risk mitigation: Prevent costly automated errors
- ⚖️ Compliance: Meet regulatory requirements (finance, healthcare)
- 💼 Trust: Users understand decisions are reviewed
- ⏱️ Efficiency: 80% fully automated + 20% manual = best of both
- 📊 Improvement: Learn from human rejections to improve AI


## 🔍 Part 1: Approval Gates

<div class="approval-box">
<strong>Approval Gate:</strong> A workflow pause point where a human must review and approve/reject before continuing.
</div>

### Fully Automated (Fast but Risky)

```
Request → Evaluate → Approve → Execute → Done
(no checks)       (instant)

Risk: Wrong decisions go live immediately
```

### With Approval Gate (Safer)

```
Request → Evaluate → 🔴 PAUSE → Human Reviews
                                    ↓
                            Approve / Reject / Ask Questions
                                    ↓
                              Execute / Cancel

Key benefit: Catch problems before they impact users
```

```python
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional
import uuid

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_INFO = "needs_info"

class ApprovalRequest:
    """Represents a workflow pause awaiting human approval."""
    
    def __init__(self, action: str, details: Dict[str, Any], requester: str):
        self.id = str(uuid.uuid4())[:8]
        self.action = action
        self.details = details
        self.requester = requester
        self.status = ApprovalStatus.PENDING
        self.created_at = datetime.now()
        self.approved_at: Optional[datetime] = None
        self.approver: Optional[str] = None
        self.comments: list[str] = []
        self.sla_deadline = self.created_at + timedelta(hours=24)
    
    def approve(self, approver: str, comment: str = ""):
        self.status = ApprovalStatus.APPROVED
        self.approver = approver
        self.approved_at = datetime.now()
        if comment:
            self.comments.append(f"✅ {approver}: {comment}")
    
    def reject(self, approver: str, reason: str):
        self.status = ApprovalStatus.REJECTED
        self.approver = approver
        self.approved_at = datetime.now()
        self.comments.append(f"❌ {approver}: {reason}")
    
    def request_info(self, approver: str, question: str):
        self.status = ApprovalStatus.NEEDS_INFO
        self.approver = approver
        self.comments.append(f"❓ {approver}: {question}")
    
    def is_sla_breached(self) -> bool:
        return datetime.now() > self.sla_deadline
    
    def time_pending(self) -> str:
        delta = datetime.now() - self.created_at
        mins = delta.total_seconds() / 60
        if mins < 60:
            return f"{int(mins)}m"
        return f"{int(mins/60)}h {int(mins%60)}m"

# Demo: Approval workflow
print("🔴 APPROVAL GATE DEMO")
print("="*70)

# 1. Workflow generates action to approve
request = ApprovalRequest(
    action="charge_customer_card",
    details={
        "customer": "Acme Corp",
        "amount": "$50,000",
        "reason": "Monthly subscription (unusual amount)",
        "risk_score": 0.87  # High risk
    },
    requester="billing_agent"
)

print(f"\n1️⃣ WORKFLOW PAUSES FOR APPROVAL")
print(f"   Request ID: {request.id}")
print(f"   Action: {request.action}")
print(f"   Details: {request.details}")
print(f"   Status: {request.status.value}")
print(f"   SLA Deadline: {request.sla_deadline.strftime('%Y-%m-%d %H:%M')}")

print(f"\n2️⃣ HUMAN REVIEWS")
print(f"   ⏳ Waiting for approval...")

print(f"\n3️⃣ HUMAN PROVIDES FEEDBACK")
request.request_info("alice@company.com", "Is this customer known to us? Check previous orders.")
print(f"   Alice asks: {request.comments[-1]}")
print(f"   Status: {request.status.value}")

print(f"\n4️⃣ WORKFLOW PROVIDES INFO")
request.comments.append(f"ℹ️ System: Found 50+ previous orders, total value $2M+")
print(f"   {request.comments[-1]}")

print(f"\n5️⃣ HUMAN APPROVES")
request.approve("alice@company.com", "Customer is trusted. Proceed.")
print(f"   {request.comments[-1]}")
print(f"   Status: {request.status.value}")
print(f"   Time pending: {request.time_pending()}")

print(f"\n✅ WORKFLOW CONTINUES")
print(f"   Execute action: charge_customer_card")
print(f"   Result: Transaction successful")
```

## 🤝 Part 2: Human Feedback Integration

<div class="human-box">
<strong>Feedback Loop:</strong> Humans provide guidance that shapes workflow behavior.
</div>

### Patterns of Human Feedback

**1. Validation:** "Is your decision correct?"
```
AI decides → Human validates → Proceed or reconsider
```

**2. Steering:** "Try this approach instead"
```
AI tries approach A → Human suggests B → AI re-runs with B
```

**3. Escalation:** "This needs senior review"
```
AI + Local reviewer → Complex case → Escalate to director
```

```python
# Multi-level approval workflow

class ApprovalWorkflow:
    """Simulate a workflow with escalation rules."""
    
    def __init__(self):
        self.requests: Dict[str, ApprovalRequest] = {}
        self.history: list[str] = []
    
    def submit_for_approval(self, request: ApprovalRequest):
        self.requests[request.id] = request
        self.history.append(f"📤 Submitted: {request.id} ({request.action})")
    
    def escalate_if_needed(self, request: ApprovalRequest, reason: str) -> bool:
        """Check if request should escalate to higher authority."""
        # Escalate if high risk or high value
        if request.details.get("risk_score", 0) > 0.8:
            self.history.append(f"🔺 Escalated: {reason} (risk_score={request.details['risk_score']})")
            return True
        if "$" in str(request.details.get("amount", "")):
            amount = float(request.details["amount"].replace("$", "").replace(",", ""))
            if amount > 100000:
                self.history.append(f"🔺 Escalated: High value transaction (${amount:,.0f})")
                return True
        return False
    
    def get_status(self):
        return self.history

# Demo: Multi-level approval
print("\n🤝 MULTI-LEVEL APPROVAL WORKFLOW")
print("="*70)

workflow = ApprovalWorkflow()

# Create diverse requests
requests_to_process = [
    ApprovalRequest(
        action="grant_refund",
        details={"amount": "$50", "reason": "defective product", "risk_score": 0.2},
        requester="support_agent"
    ),
    ApprovalRequest(
        action="grant_refund",
        details={"amount": "$250000", "reason": "bulk order", "risk_score": 0.9},
        requester="sales_agent"
    ),
]

for req in requests_to_process:
    workflow.submit_for_approval(req)
    
    if workflow.escalate_if_needed(req, "High-risk or high-value transaction"):
        print(f"\n📌 Request {req.id}:")
        print(f"   Action: {req.action}")
        print(f"   Amount: {req.details.get('amount', 'N/A')}")
        print(f"   Risk: {req.details.get('risk_score', 0):.1f}")
        print(f"   ⚠️ ESCALATED TO SENIOR REVIEWER")
    else:
        req.approve("auto_approver", "Auto-approved")
        print(f"\n📌 Request {req.id}: ✅ AUTO-APPROVED")

print("\n" + "="*70)
print("📊 WORKFLOW HISTORY:")
for item in workflow.get_status():
    print(f"  {item}")
```

## 🎯 Part 3: Conditional Routing

### Decision Tree Based on Human Input

```
Request
  ↓
AI Evaluates
  ↓
🔴 Human Reviews
  ├─ ✅ Approved → Execute
  ├─ ❌ Rejected → Cancel + Notify
  ├─ ❓ Needs Info → Request Details → Re-evaluate
  └─ 🔺 Escalated → Send to Senior → Their Decision
```

```python
# Conditional routing based on approval decision

class DecisionRouter:
    """Route workflow based on approval decision."""
    
    @staticmethod
    def route(approval: ApprovalRequest) -> str:
        """Determine next action based on approval status."""
        
        if approval.status == ApprovalStatus.APPROVED:
            return "execute_action"
        
        elif approval.status == ApprovalStatus.REJECTED:
            return "cancel_and_notify"
        
        elif approval.status == ApprovalStatus.NEEDS_INFO:
            return "request_more_info"
        
        else:
            return "wait_for_decision"

# Demo: Routing
print("\n🛤️ CONDITIONAL ROUTING")
print("="*70)

# Example 1: Approved
req1 = ApprovalRequest("process_order", {"order_id": "ORD123"}, "system")
req1.approve("admin", "Looks good")
print(f"\n📤 Decision: {req1.status.value}")
print(f"   Next step: {DecisionRouter.route(req1)} → Process order immediately")

# Example 2: Rejected
req2 = ApprovalRequest("refund_request", {"amount": "$10000", "count": 50}, "system")
req2.reject("manager", "Too many refunds this week")
print(f"\n📤 Decision: {req2.status.value}")
print(f"   Next step: {DecisionRouter.route(req2)} → Notify customer of rejection")

# Example 3: Needs info
req3 = ApprovalRequest("account_merge", {"accounts": 2}, "system")
req3.request_info("security_team", "Same person owns both accounts?")
print(f"\n📤 Decision: {req3.status.value}")
print(f"   Next step: {DecisionRouter.route(req3)} → Collect additional information")
```
