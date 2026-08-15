# Week 2 — Task Management & Dependencies

**Course:** CrewAI for Multi-Agent Systems
**Week Focus:** Master task orchestration, dependencies, priorities, and delegation to coordinate complex multi-agent workflows.

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
- Design complex task dependency graphs with 20+ interconnected tasks
- Implement parallel and sequential task execution patterns
- Manage task priorities, deadlines, and critical paths
- Handle task delegation between agents intelligently
- Build robust error handling and retry mechanisms
- Create a real-world product launch coordination system

## 📊 Real-World Context

**The Challenge:** Coordinating a product launch involves managing massive complexity:
- 25+ tasks across 6 different teams (Engineering, Marketing, Sales, Support, Legal, Product)
- Complex dependencies (can't launch marketing campaigns before product ships)
- Parallel workstreams (sales training and support docs can happen simultaneously)
- Critical path items that block everything else (legal compliance, security audit)
- High stakes: $500K/week revenue loss for each week of launch delay
- Coordination overhead: 20+ hours/week in status meetings without automation

**The Solution:** A multi-agent task orchestration crew:
1. **Product Manager Agent**: Coordinates overall launch timeline and dependencies
2. **Engineering Coordinator**: Manages technical tasks (dev, QA, deployment)
3. **Marketing Coordinator**: Handles campaigns, content, press
4. **Sales Coordinator**: Prepares training, demos, pricing
5. **Support Coordinator**: Creates docs, runbooks, FAQs
6. **Legal Coordinator**: Ensures compliance, reviews contracts

**Business Impact:**
- ⏱️ Reduce launch coordination from 6 weeks → 3 weeks
- 💰 Prevent costly delays worth $500K/week in lost revenue
- 📈 Increase visibility: all stakeholders see real-time status
- ✨ Ensure zero dependency misses with automated tracking
- 🎯 Hit 95% of milestones on-time (vs 60% manual coordination)

Companies like **Asana, Monday.com, Linear, Atlassian** use similar orchestration engines in their products.


## 🔍 Part 1: Understanding Task Structure

### Task Anatomy — The Building Blocks

Every CrewAI Task has these key components:

1. **Description**: Detailed instructions for what the agent should do
2. **Expected Output**: Specific, measurable success criteria
3. **Agent**: The agent responsible for executing this task
4. **Context**: List of prerequisite tasks (dependencies)
5. **Async Execution**: Whether task can run in parallel
6. **Callback**: Function to call when task completes
7. **Tools**: Specific tools available for this task

**Why Precision Matters:**

Poor task definition:
```
description: "Do marketing stuff"
expected_output: "Something good"
```

Good task definition:
```
description: "Create email campaign for product launch targeting enterprise customers. Include subject lines, body copy for 3 emails (announcement, features deep-dive, case study), and CTA buttons."
expected_output: "Email campaign package with: (1) 5 subject line options A/B tested, (2) 3 email templates in HTML, (3) Campaign schedule with send dates, (4) Target segment criteria, (5) Success metrics (open rate >25%, CTR >5%)"
```

The second version gives the agent clear guidance and measurable success criteria.

```python
# Installation (uncomment if needed)
# !pip install crewai crewai-tools langchain-community

from crewai import Agent, Task, Crew, Process
from langchain.llms.fake import FakeListLLM

# Create demonstration agents
demo_llm = FakeListLLM(responses=[
    "Data collection complete. Gathered Q4 sales data from all regions.",
    "Analysis complete. Top 3 regions identified with growth rates.",
    "Report generated with visualizations and insights."
])

# Simple agent for demonstration
analyst = Agent(
    role="Data Analyst",
    goal="Analyze sales data and generate insights",
    backstory="You are an experienced data analyst with expertise in sales analytics and visualization.",
    llm=demo_llm,
    verbose=True
)

# Example 1: Basic Task (no dependencies)
task1_gather_data = Task(
    description='''Collect Q4 2024 sales data from all regions.

    Data should include:
    - Region name
    - Monthly revenue (Oct, Nov, Dec)
    - Number of deals closed
    - Average deal size

    Export as CSV format.''',

    expected_output='''CSV file with columns: region, month, revenue, deals_closed, avg_deal_size.
    Should contain data for at least 5 regions covering Q4 2024.''',

    agent=analyst
)

print("✅ Task 1 Created: Data Collection")
print(f"   Description: {task1_gather_data.description[:60]}...")
print(f"   Expected Output: {task1_gather_data.expected_output[:60]}...")
print(f"   Agent: {task1_gather_data.agent.role}")
print()

# Example 2: Dependent Task (requires task1 output)
task2_analyze = Task(
    description='''Analyze the Q4 sales data to identify top-performing regions.

    Analysis should include:
    - Rank regions by total revenue
    - Calculate quarter-over-quarter growth rates
    - Identify trends and patterns
    - Flag any anomalies or concerns''',

    expected_output='''Analysis report containing:
    1. Top 3 regions by revenue with specific dollar amounts
    2. Growth rates for each region (percentage)
    3. Key trends observed
    4. Any red flags or concerns
    Format as structured text with clear sections.''',

    agent=analyst,
    context=[task1_gather_data]  # This task DEPENDS on task1
)

print("✅ Task 2 Created: Data Analysis")
print(f"   Description: {task2_analyze.description[:60]}...")
print(f"   Depends on: {len(task2_analyze.context)} task(s)")
print(f"   Context tasks: {[t.description[:30] for t in task2_analyze.context]}")
print()

# Example 3: Final Task (requires task2)
task3_report = Task(
    description='''Create executive summary report with visualizations.

    Report should include:
    - One-page executive summary
    - Key metrics dashboard
    - Regional performance charts
    - Recommendations for Q1 2025''',

    expected_output='''Executive report package with:
    - Executive summary (1 page, bullet points)
    - 3 data visualizations (described)
    - Top 3 recommendations with rationale
    - Projected Q1 targets based on Q4 performance''',

    agent=analyst,
    context=[task1_gather_data, task2_analyze]  # Depends on BOTH previous tasks
)

print("✅ Task 3 Created: Report Generation")
print(f"   Description: {task3_report.description[:60]}...")
print(f"   Depends on: {len(task3_report.context)} task(s)")
print()

print("📊 Task Dependency Chain:")
print("   Task 1 (Gather Data)")
print("      ↓")
print("   Task 2 (Analyze) ← uses Task 1 output")
print("      ↓")
print("   Task 3 (Report) ← uses Task 1 + Task 2 output")
print()
print("💡 Tasks will execute in order: 1 → 2 → 3")
print("💡 Each task receives output from its context tasks as input")
```

### Understanding Task Context

The `context` parameter is crucial for building complex workflows. Here's how it works:

**How Context Flows:**

```python
task_a = Task(description="Research topic X", ...)
task_b = Task(description="Write about topic X", context=[task_a], ...)
```

When `task_b` executes:
1. CrewAI automatically passes `task_a`'s output to `task_b`
2. The agent performing `task_b` can reference this information
3. No manual output passing required!

**Multiple Dependencies:**

```python
research_task = Task(description="Research...", ...)
data_task = Task(description="Gather data...", ...)
write_task = Task(
    description="Write article using research and data",
    context=[research_task, data_task],  # Both outputs available!
    ...
)
```

**Best Practices:**
- List dependencies in `context` even if they seem obvious
- Order matters: list them in the order they should be considered
- Don't create circular dependencies (A depends on B depends on A)
- Keep dependency chains reasonable (max 3-4 levels deep)

```python
# Demonstrate context flow with realistic example

from crewai import Agent, Task, Crew, Process
from langchain.llms.fake import FakeListLLM

# Create specialized agents
research_llm = FakeListLLM(responses=[
    '''MARKET RESEARCH FINDINGS:
- Global project management software market: $6.8B (2024)
- Expected growth: 15% CAGR through 2029
- Key trends: AI automation, remote collaboration, integration ecosystems
- Top competitors: Asana, Monday.com, ClickUp, Linear
- Customer pain points: Too complex UI, poor integrations, expensive enterprise plans
- Opportunity: Mid-market segment ($500-5K/month) is underserved'''
])

writer_llm = FakeListLLM(responses=[
    '''POSITIONING STRATEGY:

Based on market research, our positioning should be:

**Target Segment**: Mid-market teams (50-200 people)
**Key Differentiator**: AI-powered automation WITHOUT complexity
**Positioning Statement**: "The intelligent project management platform that automates busywork so teams can focus on what matters"

**Messaging Pillars:**
1. Simple AI: Automation that just works, no setup required
2. Team-First: Built for collaboration, not solo management
3. Fair Pricing: Enterprise features at mid-market prices

**Competitive Advantages:**
- vs Asana: Smarter automation, better price
- vs Monday.com: Cleaner UI, faster setup
- vs ClickUp: More focused, less overwhelming

**Launch Narrative**:
Focus on "AI that saves time" rather than "AI that's powerful". Show real time savings (2hrs/week per team member) not abstract capabilities.'''
])

researcher = Agent(
    role="Senior Market Research Analyst",
    goal="Conduct competitive analysis and identify market opportunities",
    backstory="10 years experience in SaaS market research. Expert at identifying gaps and positioning opportunities.",
    llm=research_llm,
    verbose=True
)

strategist = Agent(
    role="Product Marketing Strategist",
    goal="Develop positioning and messaging based on market insights",
    backstory="Former product marketer at high-growth startups. Skilled at translating research into compelling positioning.",
    llm=writer_llm,
    verbose=True
)

# Task 1: Research (no dependencies)
research_task = Task(
    description='''Conduct market research for our new project management tool launch.

    Research should cover:
    1. Market size and growth projections
    2. Competitive landscape (top 5 competitors)
    3. Customer pain points and unmet needs
    4. Pricing trends and expectations
    5. Technology trends (AI, integrations, mobile)

    Focus on mid-market segment (50-200 person teams).''',

    expected_output='''Market research report with:
    - Market size with growth projections
    - Competitor analysis (features, pricing, positioning)
    - List of top 3 customer pain points
    - Market opportunities and gaps
    - Recommended target segment''',

    agent=researcher
)

# Task 2: Strategy (depends on research)
strategy_task = Task(
    description='''Using the market research, develop our product positioning and messaging strategy.

    Strategy should define:
    1. Target customer segment (be specific)
    2. Unique value proposition
    3. Key differentiators vs competitors
    4. Messaging pillars (3-5 core themes)
    5. Launch narrative and story

    Make this actionable for the marketing team.''',

    expected_output='''Positioning strategy document with:
    - Target segment description
    - Positioning statement (one sentence)
    - 3-5 messaging pillars with explanations
    - Competitive differentiation matrix
    - Recommended launch narrative
    All based on research findings.''',

    agent=strategist,
    context=[research_task]  # DEPENDS on research findings
)

# Create crew to execute
positioning_crew = Crew(
    agents=[researcher, strategist],
    tasks=[research_task, strategy_task],
    process=Process.sequential,
    verbose=True
)

print("🎯 Executing Positioning Strategy Workflow...")
print("="*70)
print()

# Run the crew
result = positioning_crew.kickoff()

print()
print("="*70)
print("✅ Workflow Complete!")
print()
print("📋 Final Strategy:")
print(result)
print()
print("💡 Notice how the strategy task used research task output automatically!")
```

## 📝 Part 2: Task Dependencies and Graphs

In complex workflows, you'll have multiple dependency paths. Understanding how to model these is crucial.

## Week 2 Project: Event Planning Crew

**Build a complete event planning coordination system with 20+ tasks.**

### Scenario:
Coordinate a major tech conference with 500+ attendees.

### Requirements:

**6 Agents:**
1. Venue Coordinator
2. Speakers Coordinator
3. Marketing Manager
4. Logistics Manager
5. Sponsorship Manager
6. Technology Coordinator

**20+ Tasks including:**
- Venue Tasks: Book venue, negotiate contract, plan layout
- Speaker Tasks: Recruit speakers, collect bios/photos, schedule talks
- Marketing Tasks: Website, email campaigns, social media
- Logistics Tasks: Catering, A/V equipment, registration
- Sponsorship Tasks: Recruit sponsors, create packages, manage deliverables
- Technology Tasks: Event app, livestreaming, WiFi

**Dependencies:**
- Marketing can't launch until speakers confirmed
- Logistics depends on final attendee count
- Technology setup depends on venue layout

**Deliverables:**
- Complete task dependency graph
- All agents with detailed backstories
- 20+ tasks with clear dependencies
- Crew that executes the plan
- Timeline showing parallel vs sequential work

### Starter Code:

```python
# Week 2 Project Starter

from crewai import Agent, Task, Crew, Process
from langchain.llms.fake import FakeListLLM

# TODO: Create 6 specialized agents
# venue_coordinator = Agent(...)
# speakers_coordinator = Agent(...)
# marketing_manager = Agent(...)
# logistics_manager = Agent(...)
# sponsorship_manager = Agent(...)
# technology_coordinator = Agent(...)

# TODO: Create 20+ tasks with dependencies
# Example structure:
# task_venue_search = Task(description="Research and evaluate 5 potential venues...", ...)
# task_venue_book = Task(description="Negotiate and book chosen venue...", context=[task_venue_search], ...)
# task_speaker_recruit = Task(description="Recruit 15 industry speakers...", ...)
# task_marketing_site = Task(description="Build event website...", context=[task_speaker_recruit, task_venue_book], ...)

# TODO: Create crew and execute
# event_crew = Crew(agents=[...], tasks=[...], process=Process.sequential)
# result = event_crew.kickoff()

print("🎯 Your implementation here!")
print("Build a complete 20+ task event planning system")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Task orchestration:**
- Task components: description, expected output, context
- Dependency management via context parameter
- Sequential vs parallel execution patterns

✅ **Complex workflows:**
- Multi-level dependency chains
- Critical path identification
- Parallel workstream coordination

✅ **Best practices:**
- Clear expected outputs prevent confusion
- Explicit dependencies ensure correct order
- Async tasks enable parallelization
- Error handling prevents workflow failures

✅ **Real-world application:**
- Built product launch coordination system
- Managed 25+ task dependencies
- Implemented parallel and sequential execution

## 🔜 Next Week: Team Collaboration

In Week 3, we'll master **agent collaboration**:
- Inter-agent communication protocols
- Consensus building and voting
- Conflict resolution strategies
- Knowledge sharing between agents
- Building software development teams

**Preview question:** How would you handle disagreement between a QA agent and a Developer agent about whether a bug is critical?

## 📚 Additional Resources

- [CrewAI Task Documentation](https://docs.crewai.com/core-concepts/Tasks/)
- [Process Types](https://docs.crewai.com/core-concepts/Processes/)
- [Async Execution Patterns](https://docs.crewai.com/how-to/async-execution/)

---

**🎉 Congratulations on completing Week 2!** You can now orchestrate complex multi-agent workflows with dependencies. See you next week!
