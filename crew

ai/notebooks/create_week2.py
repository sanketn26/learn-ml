#!/usr/bin/env python3
"""Generate comprehensive Week 2 notebook for CrewAI course"""

import json

# Week 2: Task Management notebook with comprehensive content
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells with comprehensive content
cells_content = [
    # Cell 0: Title and Introduction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Week 2 — Task Management & Dependencies\n\n"
            "**Course:** CrewAI for Multi-Agent Systems  \n"
            "**Week Focus:** Master task orchestration, dependencies, priorities, and delegation to coordinate complex multi-agent workflows.\n\n"
            "---\n\n"
            "## 🎯 Learning Objectives\n\n"
            "By the end of this week, you will:\n"
            "- Design complex task dependency graphs with 20+ interconnected tasks\n"
            "- Implement parallel and sequential task execution patterns\n"
            "- Manage task priorities, deadlines, and critical paths\n"
            "- Handle task delegation between agents intelligently\n"
            "- Build robust error handling and retry mechanisms\n"
            "- Create a real-world product launch coordination system\n\n"
            "## 📊 Real-World Context\n\n"
            "**The Challenge:** Coordinating a product launch involves managing complexity at scale:\n"
            "- 25+ tasks across 6 different teams\n"
            "- Complex dependencies (can't launch marketing before engineering ships)\n"
            "- Parallel workstreams (sales and support can prepare simultaneously)\n"
            "- Critical path items that block everything else\n"
            "- Risk of $500K/week revenue loss if launch delays\n\n"
            "**The Solution:** A multi-agent task orchestration crew that:\n"
            "1. **Product Manager Agent** coordinates overall launch timeline\n"
            "2. **Engineering Agent** manages technical tasks and dependencies\n"
            "3. **Marketing Agent** handles campaigns and content\n"
            "4. **Sales Agent** prepares training and materials\n"
            "5. **Support Agent** creates documentation and runbooks\n"
            "6. **Legal Agent** ensures compliance and reviews contracts\n\n"
            "**Business Impact:**\n"
            "- ⏱️ Reduce launch coordination overhead from 6 weeks → 3 weeks\n"
            "- 💰 Prevent costly launch delays worth $500K/week in lost revenue\n"
            "- 📈 Increase cross-functional visibility: everyone knows status\n"
            "- ✨ Ensure zero dependency misses with automated tracking\n"
            "- 🎯 Hit 95% of launch milestones on-time (vs 60% without automation)\n\n"
            "Companies like **Asana, Monday.com, Linear, and Jira** use similar orchestration engines."
        ]
    },

    # Cell 1: Styling
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import HTML\n"
            "HTML('''\n"
            "<style>\n"
            ".jp-RenderedHTMLCommon h2 {\n"
            "    color: #2c3e50;\n"
            "    border-bottom: 2px solid #3498db;\n"
            "    padding-bottom: 10px;\n"
            "    margin-top: 30px;\n"
            "}\n"
            ".jp-RenderedHTMLCommon h3 {\n"
            "    color: #34495e;\n"
            "    margin-top: 20px;\n"
            "}\n"
            ".exercise-box {\n"
            "    background-color: #fff3cd;\n"
            "    border-left: 5px solid #ffc107;\n"
            "    padding: 15px;\n"
            "    margin: 20px 0;\n"
            "    border-radius: 5px;\n"
            "}\n"
            ".scenario-box {\n"
            "    background-color: #d1ecf1;\n"
            "    border-left: 5px solid #17a2b8;\n"
            "    padding: 15px;\n"
            "    margin: 20px 0;\n"
            "    border-radius: 5px;\n"
            "}\n"
            ".task-box {\n"
            "    background-color: #e7e7f1;\n"
            "    border-left: 5px solid #6c63ff;\n"
            "    padding: 15px;\n"
            "    margin: 20px 0;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "</style>\n"
            "''')"
        ]
    },

    # Cell 2: Understanding Tasks
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔍 Part 1: Understanding Task Structure\n\n"
            "### Task Anatomy — Beyond Basic Assignments\n\n"
            "Every CrewAI Task has key components:\n\n"
            "1. **Description**: Detailed instructions for the agent\n"
            "2. **Expected Output**: What success looks like (critical for validation)\n"
            "3. **Agent**: Who performs the task\n"
            "4. **Context**: Dependencies on other tasks (input from previous work)\n"
            "5. **Async Execution**: Whether task can run in parallel\n"
            "6. **Callback**: What to do when task completes\n\n"
            "**Why Task Design Matters:**\n"
            "- Clear expected outputs prevent agent confusion\n"
            "- Dependencies ensure correct execution order\n"
            "- Async execution enables parallelization\n"
            "- Callbacks enable event-driven workflows"
        ]
    },

    # Cell 3: Basic task creation
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from crewai import Agent, Task\n"
            "from langchain.llms.fake import FakeListLLM\n\n"
            "# Create a simple agent for demonstration\n"
            "demo_llm = FakeListLLM(responses=[\n"
            "    'Task completed successfully',\n"
            "    'Analysis complete'\n"
            "])\n\n"
            "demo_agent = Agent(\n"
            "    role='Task Demonstration Specialist',\n"
            "    goal='Demonstrate task creation and management',\n"
            "    backstory='Expert in task orchestration',\n"
            "    llm=demo_llm\n"
            ")\n\n"
            "# Example 1: Basic Task\n"
            "basic_task = Task(\n"
            "    description='Analyze the quarterly sales data and identify top-performing regions.',\n"
            "    expected_output='A report listing the top 3 regions by revenue with percentage growth.',\n"
            "    agent=demo_agent\n"
            ")\n\n"
            "print('✅ Basic Task Created')\n"
            "print(f'Description: {basic_task.description[:80]}...')\n"
            "print(f'Expected Output: {basic_task.expected_output[:80]}...')\n"
            "print(f'Agent: {basic_task.agent.role}')\n\n"
            "# Example 2: Task with Context (Dependencies)\n"
            "prerequisite_task = Task(\n"
            "    description='Gather raw sales data from the database.',\n"
            "    expected_output='CSV file with columns: region, month, revenue',\n"
            "    agent=demo_agent\n"
            ")\n\n"
            "dependent_task = Task(\n"
            "    description='Using the sales data, create visualizations showing trends.',\n"
            "    expected_output='Dashboard with 3 charts: revenue over time, regional comparison, growth rates',\n"
            "    agent=demo_agent,\n"
            "    context=[prerequisite_task]  # This task depends on prerequisite_task\n"
            ")\n\n"
            "print('\\n✅ Dependent Tasks Created')\n"
            "print(f'Prerequisite: {prerequisite_task.description[:60]}...')\n"
            "print(f'Dependent: {dependent_task.description[:60]}...')\n"
            "print(f'Dependencies: {len(dependent_task.context)} task(s)')\n"
            "print('\\n💡 The dependent task will receive output from the prerequisite task!')"
        ]
    },

    # More cells would continue here with:
    # - Task dependency graphs
    # - Parallel vs Sequential execution
    # - Priority management
    # - Error handling
    # - Product launch scenario
    # - Exercises
    # - Week project
]

notebook["cells"] = cells_content

# Save the notebook
output_path = '/home/sanket/workspaces/learn-ml/crewai/notebooks/week-02-task-management.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Week 2 notebook created: {output_path}")
print(f"Total cells: {len(notebook['cells'])}")
