# Exercises — Week 2 — Complex Workflows: Subgraphs & Conditional Routing

Do these after reading [Week 2 — Complex Workflows: Subgraphs & Conditional Routing](../week-02.md).

## Hands-On Exercises

### Exercise 1: Conditional Routing
Build a workflow that routes requests based on priority:
- High priority → expedited path (2 reviewers)
- Normal priority → standard path (1 reviewer)
- Low priority → async processing

### Exercise 2: Parallel Tasks
Implement customer onboarding with parallel steps:
- Send welcome email
- Create dashboard
- Assign account manager
- All run simultaneously, then notify operations

```python
# TODO: Build conditional routing workflow
# TODO: Implement parallel execution pattern
# TODO: Create reusable subgraph
# TODO: Add retry logic with exponential backoff
```

## Assignment: Multi-Branch Workflow

**Build:** Customer support routing system

**Features:**
1. **Conditional routing:**
   - Technical issues → engineering team
   - Billing issues → finance team
   - Feature requests → product team

2. **Parallel operations:**
   - Send auto-response
   - Create ticket
   - Notify team
   - Update knowledge base

3. **Dynamic escalation:**
   - High-urgency → immediate escalation
   - Old unresolved → escalate to manager
   - Customer is VIP → priority handling

## Key Takeaways

✅ Conditional routing based on state  
✅ Parallel execution patterns  
✅ Subgraphs for modularity  
✅ Retry logic with backoff  
✅ Dynamic workflow structure  
✅ Error handling strategies  

## 🔜 Next Week: Persistence & Replay

Save workflow state, resume from checkpoints, replay for debugging
