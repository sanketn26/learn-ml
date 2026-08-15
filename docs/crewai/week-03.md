# Week 3 — Team Collaboration & Communication

**Course:** CrewAI for Multi-Agent Systems  
**Week Focus:** Master inter-agent communication, delegation strategies, and consensus-building for complex collaborative workflows.

---

## If you already write software

CrewAI is **jobs + role descriptions + a shared ticket queue**. An “agent” is a worker with a system prompt (its job description), tools (its IAM permissions), and a task (a ticket). A “crew” is the team you put on the sprint.

```
Job description          agent role + backstory + goal
IAM policy               tools the agent is allowed to call
Ticket                   Task(description, expected_output, agent=)
Sprint team              Crew(agents, tasks, process=sequential|hierarchical)
Standup notes            the shared result / memory the next task reads
```

This is useful when the work actually needs different roles (researcher vs writer vs reviewer). It is ceremony when one function would do. Staff a crew the way you staff a project: few roles, clear tickets, an output contract. “Add more agents” is not an architecture.

## 🎯 Learning Objectives

By the end of this week, you will:
- Implement agent delegation and role-based communication
- Build consensus mechanisms for disagreeing agents
- Create hierarchical team structures (CEO, leads, specialists)
- Handle agent conflicts and decision-making
- Implement feedback loops and iterative refinement
- Build a real-world software development team

## 📊 Real-World Context

**The Challenge:** Building software requires multiple perspectives:
- **Product Manager**: "This feature must do X"
- **Architect**: "We need microservices architecture"
- **Security Lead**: "We must add authentication"
- **Dev Lead**: "We have 2 weeks for this"
- **QA Lead**: "We need 95% test coverage"

**Without collaboration:**
- Team works in silos (no alignment)
- Conflicting requirements cause delays
- Poor quality (no peer review)
- Missed security/performance considerations

**With AI Team Collaboration:**
1. **Communication**: Agents discuss requirements and constraints
2. **Consensus**: Reach agreement on technical approach
3. **Specialization**: Each agent contributes expertise
4. **Conflict Resolution**: Handle disagreements systematically
5. **Iteration**: Refine decisions based on feedback

**Business Impact:**
- ⏱️ Reduce design review meetings from 20 hrs → 2 hrs
- ✅ Catch 40% more issues (security, architecture, performance)
- 📈 Faster time-to-market (better decisions = fewer rewrites)
- 💰 Save $200K/year in review/rework cycles

Companies like **Google, GitHub, Anthropic** use collaborative agent patterns internally.


## 🔍 Part 1: Communication Patterns

### Agent Communication Types

**Direct Communication:**
```
Agent A → Task Output → Agent B
         (context parameter)
```
Agent B uses Agent A's output as input.

**Via Shared State:**
```
Agent A → Update Database ← Agent B
         ↓
      Central Repository
```
Agents write and read from shared knowledge base.

**Hierarchical (CEO Pattern):**
```
         ┌─ Product Agent
         │
    [CEO Agent] ─ Architecture Agent
         │
         └─ Security Agent

CEO makes final decisions after hearing from all specialists.
```

**Peer Review (Consensus):**
```
Agent A suggests solution → All agents vote
                         ↓
                    Consensus reached or
                    Escalate to CEO
```

## 📚 Part 2: Delegation Strategies

### 2.1 When to Delegate

**Delegate if:**
- Another agent has specific expertise
- Task requires specialized knowledge
- Parallel execution possible
- Subtask is well-defined

**Don't delegate if:**
- Task requires real-time feedback
- Decision is critical (CEO should decide)
- Agent would just repeat your work
- Output isn't clear (ambiguous task)

### 2.2 Effective Delegation Pattern

```
1. CLARIFY: "Here's what I need and why"
2. SPECIFY: "Here are detailed requirements"
3. DELEGATE: "Task: {task}, Context: {context}, Success criteria: {criteria}"
4. RECEIVE: Get output from delegated agent
5. REVIEW: Check if output meets criteria
6. ITERATE: If not satisfied, provide feedback and retry
```

```python
from crewai import Agent, Task, Crew, Process
from langchain.llms.fake import FakeListLLM

# Create specialized agents
product_llm = FakeListLLM(responses=[
    "From product perspective: Users need real-time notifications, offline support, and mobile app. Core requirement: notification system with 99.9% uptime SLA."
])

architect_llm = FakeListLLM(responses=[
    "Architecture recommendation: Microservices (API, Notifications, Auth, Analytics). Use Kafka for async messaging. RDS for persistence. Redis for caching."
])

security_llm = FakeListLLM(responses=[
    "Security requirements: OAuth2 for auth, TLS 1.3 for comms, encrypt PII at rest, SOC2 compliance, rate limiting, API key rotation."
])

# Product Manager Agent
product_agent = Agent(
    role="Product Manager",
    goal="Define user requirements and business success criteria",
    backstory="Seasoned PM with 8 years at Stripe. Obsessed with user needs and product-market fit.",
    llm=product_llm,
    verbose=True
)

# Architect Agent
architect_agent = Agent(
    role="Technical Architect",
    goal="Design scalable, maintainable technical architecture",
    backstory="Former Netflix infrastructure engineer. Expert in microservices and cloud systems.",
    llm=architect_llm,
    verbose=True
)

# Security Lead Agent
security_agent = Agent(
    role="Security Lead",
    goal="Ensure system meets enterprise security requirements",
    backstory="Security researcher with 10 years experience. Led security at Amazon. Paranoid about attacks.",
    llm=security_llm,
    verbose=True
)

print("✅ Created specialized agents:")
print(f"  1. {product_agent.role}")
print(f"  2. {architect_agent.role}")
print(f"  3. {security_agent.role}")
```

## 🏗️ Part 3: Building Hierarchical Teams

### The CEO Pattern

<div class="team-box">
<strong>👔 CEO Agent:</strong> Makes final decisions after consulting specialists
</div>

CEO responsibilities:
1. **Gather Input**: Ask each specialist for their perspective
2. **Identify Conflicts**: Find areas of disagreement
3. **Facilitate Discussion**: Have specialists address each other's concerns
4. **Make Decision**: Apply business judgment to resolve conflicts
5. **Communicate**: Explain decision to team

```python
# CEO Agent coordinates the team
ceo_llm = FakeListLLM(responses=[
    """DESIGN DECISION SUMMARY:

After consulting Product, Architecture, and Security:

✅ APPROVED:
- Microservices architecture (Address, Notifications, Auth, Analytics)
- Real-time notification system with Kafka
- OAuth2 + TLS 1.3 security
- Offline support via local DB sync

⚠️ CONSTRAINTS:
- Timeline: 6 weeks (Phase 1: Core API, Phase 2: Mobile, Phase 3: Analytics)
- Budget: $80K infrastructure/month
- Team: 2 leads + 6 engineers

🎯 CRITICAL PATH:
1. Week 1-2: Auth + Notification service
2. Week 3-4: API Gateway + main services
3. Week 5-6: Mobile app + offline sync

NEXT: Dev Lead to create detailed sprint plan.
"""
])

ceo_agent = Agent(
    role="Engineering Director (CEO)",
    goal="Make architecture decisions that balance product needs, technical excellence, and security",
    backstory="Former CTO at Series B startup. Excellent at consensus-building. Cares deeply about team health.",
    llm=ceo_llm,
    verbose=True,
    allow_delegation=True
)

print("✅ CEO Agent created")
```

## 💬 Part 4: Conflict Resolution

### Common Conflicts in Teams

<div class="conflict-box">
<strong>Example Conflict:</strong><br>
<strong>Architect:</strong> "We need 3 months for proper microservices architecture"<br>
<strong>Product:</strong> "Users need this in 4 weeks or we lose competitive advantage"<br>
<strong>Security:</strong> "Rushing leads to security vulnerabilities"
</div>

### Resolution Framework

**Step 1: Understand each position**
- Architect: Quality and long-term maintainability
- Product: User needs and time-to-market
- Security: Risk mitigation

**Step 2: Find middle ground**
- Phase approach: MVP (core only) in 4 weeks, then full platform
- Security in-built from start (not added later)
- Architecture allows future scalability

**Step 3: Clear trade-offs**
- "We sacrifice some features (Phase 2) to hit market window"
- "We prioritize security even in MVP"
- "Core architecture designed for scale"

**Step 4: Document decision**
- Why this approach
- Who agrees/disagrees
- How we'll revisit if needed
