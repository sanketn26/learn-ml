# Exercises — Week 1 — Agent Fundamentals

Do these after reading [Week 1 — Agent Fundamentals](../week-01.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Create a Customer Support Team**

    Build a 3-agent customer support crew:

    - **Intake Agent:** Classifies tickets (bug/feature/question/urgent)

    - **Resolution Agent:** Provides solutions or troubleshooting steps

    - **Escalation Agent:** Determines if human intervention needed

    **Requirements:**

    - Define roles, goals, and backstories for each agent

    - Create tasks with clear expected outputs

    - Build a crew with sequential process

    - Test with a sample support ticket



```python
# Your solution here!
# Hint: Start by creating your three agents

# intake_agent = Agent(
#     role="...",
#     goal="...",
#     backstory="...",
#     llm=...
# )

# Test ticket:
test_ticket = """
Subject: Can't export data - getting timeout error
From: customer@example.com

Hi, I'm trying to export our Q4 analytics (about 150K rows) and it keeps timing out 
after 30 seconds with error ERR_TIMEOUT_500. This is blocking our board presentation tomorrow!
"""
```

!!! example "Exercise"

    **🎯 Exercise 2: Build a Code Review Team**

    Create a software development crew:

    - **Code Analyzer:** Reviews code for bugs and issues

    - **Security Auditor:** Checks for security vulnerabilities

    - **Performance Expert:** Identifies optimization opportunities

    **Challenge:** Have all three agents run in parallel, then synthesize findings.



```python
# Your solution here!
# Consider: How would you structure parallel execution?
# How would you combine results from multiple agents?

sample_code = """
def process_user_input(user_id, data):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    
    # Process all data in memory
    processed = []
    for item in data:
        processed.append(transform(item))
    
    return processed
"""
```

!!! example "Exercise"

    **🎯 Exercise 3: Add Agent Tools**

    Enhance your research agent with tools:

    - Search tool (for web search)

    - Calculator tool (for data analysis)

    - File reader tool (to access knowledge base)

    **Hint:** Use `from crewai_tools import SerperDevTool, CalculatorTool`



```python
# Your solution here!
# from crewai_tools import Tool, SerperDevTool

# enhanced_researcher = Agent(
#     role="...",
#     goal="...",
#     backstory="...",
#     tools=[...],  # Add tools here
#     llm=...
# )
```

## 📝 Week 1 Project: Build a Blog Post Creation Team

**Build a complete multi-agent blog creation system:**

### Requirements:

1. **Four Agents:**
   - Topic Researcher: Finds trending topics
   - Content Researcher: Gathers detailed information on chosen topic
   - Writer: Creates the blog post
   - Editor: Reviews and approves

2. **Four Tasks:**
   - Identify 3 trending topics with SEO potential
   - Research the selected topic comprehensively
   - Write a 1200+ word blog post
   - Edit and provide quality assessment

3. **Features:**
   - Detailed agent backstories (show personality)
   - Clear task dependencies
   - Expected outputs defined for each task
   - At least one agent with tools

4. **Testing:**
   - Run the crew on topic: "Future of Remote Work"
   - Show the complete output from each agent
   - Demonstrate agent collaboration

### Evaluation Criteria:
- ✅ All agents have distinct, well-defined roles
- ✅ Backstories create clear expertise and personality
- ✅ Tasks flow logically with dependencies
- ✅ Output shows agent collaboration
- ✅ Final blog post is comprehensive and well-structured

### Starter Code:

```python
# Week 1 Project: Blog Creation Team

from crewai import Agent, Task, Crew, Process
from langchain.llms.fake import FakeListLLM

# TODO: Create your agents
# topic_researcher = Agent(...)
# content_researcher = Agent(...)
# writer = Agent(...)
# editor = Agent(...)

# TODO: Create your tasks
# topic_task = Task(...)
# research_task = Task(...)
# writing_task = Task(...)
# editing_task = Task(...)

# TODO: Create and run crew
# blog_crew = Crew(...)
# result = blog_crew.kickoff()

# TODO: Display results
print("Your implementation here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Multi-agent architecture:**
- Why specialized agents > single generalist agent
- Agent components: role, goal, backstory, tools, config
- How agents collaborate through task dependencies

✅ **Agent design principles:**
- Specific roles create better performance
- Backstories influence behavior and output quality
- Goals should be measurable and actionable
- Tools extend agent capabilities

✅ **Collaboration patterns:**
- Sequential: Simple, predictable workflows
- Hierarchical: Manager delegates to specialists
- Parallel: Simultaneous execution for speed
- Consensus: Multiple perspectives for decisions

✅ **Real-world application:**
- Built production-grade content creation team
- Implemented research → write → edit workflow
- Demonstrated agent specialization benefits

## 🔜 Next Week: Task Management

In Week 2, we'll dive deep into **task orchestration**:
- Complex task dependencies and graphs
- Parallel vs sequential task execution
- Task priorities and delegation
- Error handling and retries
- Building a product launch coordination crew

**Preview question:** How would you coordinate 20+ tasks across 5+ agents for a product launch?

## 📚 Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [Multi-Agent Systems Research](https://arxiv.org/abs/2308.08155)
- [Agent Design Patterns](https://www.patterns.dev/posts/agents-pattern/)

---

**🎉 Congratulations on completing Week 1!** You now understand how to design and orchestrate intelligent multi-agent systems. See you next week for advanced task management! 🚀
