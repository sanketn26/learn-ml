# 🚀 AI Frameworks Learning Platform

A comprehensive, multi-course learning platform covering machine learning fundamentals, LLM orchestration, workflow automation, and multi-agent systems.

## 📚 Courses Overview

### 1. **ML Fundamentals** (12 weeks)
Applied Machine Learning for SaaS Analytics through realistic business scenarios.

- **Focus:** NumPy, Pandas, Scikit-learn, Deep Learning
- **Topics:** Data manipulation, visualization, statistical analysis, feature engineering, supervised/unsupervised learning, embeddings, model evaluation
- **Datasets:** Real-world SaaS telemetry data (50K-220K records per file)
- **Entry Point:** [`index.html`](./index.html) or [`notebooks/`](./notebooks/)

**Weeks:**
- Weeks 1-4: NumPy, Pandas, Visualization, Statistics
- Weeks 5-7: Feature Engineering, Classification, Regression
- Weeks 8-10: Clustering, Dimensionality Reduction, Anomaly Detection
- Weeks 11-12: Embeddings, Recommender Systems, Production ML

---

### 2. **LangChain Mastery** (6 weeks)
Build AI applications using LangChain's orchestration framework.

- **Focus:** LLM fundamentals, chains, agents, RAG, production deployment
- **Technology:** LangChain, OpenAI/LLMs, embeddings, vector stores
- **Progression:** Basics → Memory → Agents → RAG → Evaluation → Production
- **Entry Point:** [`langchain/index.html`](./langchain/index.html)

**Weeks:**
1. LangChain Basics — LLM fundamentals, prompts, output parsing
2. Memory & Conversation — Buffer, summary, entity, and knowledge graph memory
3. Agents & Tools — ReAct pattern, tool selection, error handling
4. RAG & Embeddings — Vector stores, FAISS, semantic search, document loading
5. Evaluation & Debugging — LangSmith, metrics, testing strategies
6. Production Deployment — Async chains, caching, Docker, monitoring

---

### 3. **LangGraph Workflows** (4 weeks)
Build complex, debuggable workflows with state graphs and deterministic routing.

- **Focus:** State graphs, conditional routing, persistence, human-in-the-loop
- **Technology:** LangGraph, state machines, workflow orchestration
- **Progression:** Basics → Workflows → Persistence → Human Approval
- **Entry Point:** [`langgraph/index.html`](./langgraph/index.html)

**Weeks:**
1. State Graphs & Basics — State schemas, nodes, edges, graph composition
2. Multi-Step Workflows — Conditional routing, loops, error handling
3. Persistence & Replay — Checkpoints, workflow history, debugging
4. Human-in-the-Loop & Production — Approval gates, deployment, monitoring

---

### 4. **Crew.ai Multi-Agents** (4 weeks)
Design and deploy autonomous agent teams that collaborate on complex tasks.

- **Focus:** Agent design, task management, team coordination, scaling
- **Technology:** Crew.ai, multi-agent orchestration, collaboration patterns
- **Progression:** Agents → Tasks → Collaboration → Production
- **Entry Point:** [`crewai/index.html`](./crewai/index.html)

**Weeks:**
1. Agent Fundamentals — Roles, goals, personalities, tool integration
2. Task Management — Task specification, dependencies, sequencing
3. Team Collaboration — Communication, parallel execution, conflict resolution
4. Production & Scaling — Deployment, cost optimization, monitoring, feedback

---

## 📁 Directory Structure

```
learn-ml/
├── index.html                          # Platform landing page (all courses)
├── README.md                           # This file
│
├── notebooks/                          # ML Course (12 weeks)
│   ├── week-01-saas.ipynb
│   ├── week-02-saas.ipynb
│   └── ... (12 weeks total)
├── docs/                               # ML Course HTML renders
│   ├── week-01-saas.html
│   └── ... (12 weeks HTML)
├── data/                               # ML Course datasets
│   ├── subscriptions.csv
│   ├── user_events.csv
│   ├── feature_usage.csv
│   ├── feedback.json
│   └── product_catalog.csv
├── assignments/                        # ML Course exercises
│
├── langchain/                          # LangChain Course
│   ├── index.html
│   ├── notebooks/                      # 6 week notebooks
│   │   ├── week-01-langchain-basics.ipynb
│   │   ├── week-02-memory-conversation.ipynb
│   │   ├── week-03-agents-tools.ipynb
│   │   ├── week-04-rag-embeddings.ipynb
│   │   ├── week-05-evaluation-debugging.ipynb
│   │   └── week-06-production-deployment.ipynb
│   ├── docs/                           # 6 week HTML renders
│   │   └── week-*.html
│   ├── data/                           # Course data files
│   └── assignments/                    # Course exercises
│
├── langgraph/                          # LangGraph Course
│   ├── index.html
│   ├── notebooks/                      # 4 week notebooks
│   │   ├── week-01-graphs-basics.ipynb
│   │   ├── week-02-workflows.ipynb
│   │   ├── week-03-persistence-replay.ipynb
│   │   └── week-04-human-in-loop.ipynb
│   ├── docs/                           # 4 week HTML renders
│   └── assignments/
│
├── crewai/                             # Crew.ai Course
│   ├── index.html
│   ├── notebooks/                      # 4 week notebooks
│   │   ├── week-01-agent-fundamentals.ipynb
│   │   ├── week-02-task-management.ipynb
│   │   ├── week-03-team-collaboration.ipynb
│   │   └── week-04-production-scaling.ipynb
│   ├── docs/                           # 4 week HTML renders
│   └── assignments/
│
└── solutions/                          # Reference solutions
```

---

## 🚀 Quick Start

### Option 1: View in Browser
Open `index.html` in any modern web browser to navigate all courses.

```bash
# From the root directory
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or
start index.html  # Windows
```

### Option 2: Run Jupyter Notebooks Locally

**Prerequisites:**
```bash
pip install jupyter numpy pandas scikit-learn matplotlib scipy
pip install langchain openai  # For LangChain course
pip install langgraph  # For LangGraph course
pip install crewai  # For Crew.ai course
```

**Launch:**
```bash
# Navigate to course directory
cd notebooks/  # or langchain/notebooks/, etc.

# Start Jupyter
jupyter notebook
```

---

## 📊 Course Statistics

| Metric | ML | LangChain | LangGraph | Crew.ai | Total |
|--------|----|-----------|-----------|---------| ------|
| **Weeks** | 12 | 6 | 4 | 4 | 26 |
| **Notebooks** | 12 | 6 | 4 | 4 | 26 |
| **HTML Files** | 12 | 6 | 4 | 4 | 26 |
| **Assignments** | 12 | - | - | - | 12+ |
| **Datasets** | 5 | - | - | - | 5+ |

---

## 🎓 Learning Path Recommendations

### Path 1: Complete Beginner
Start with **ML Fundamentals** to build foundational Python and data science skills, then explore AI frameworks.

```
ML Fundamentals (12 weeks)
  ↓
LangChain Mastery (6 weeks)
  ↓
LangGraph Workflows (4 weeks)
  ↓
Crew.ai Multi-Agents (4 weeks)
```

### Path 2: AI Application Developer
If you already know Python and ML, jump directly to **LangChain**, then explore related frameworks.

```
LangChain Mastery (6 weeks)
  ↓
LangGraph Workflows (4 weeks)
  ↓
Crew.ai Multi-Agents (4 weeks)
```

### Path 3: Workflow & Agent Specialist
If you're familiar with LLMs, focus on **LangGraph** and **Crew.ai** for advanced orchestration.

```
LangGraph Workflows (4 weeks)
  ↓
Crew.ai Multi-Agents (4 weeks)
```

---

## 💡 Each Course Includes

- **📖 Structured Notebooks** with learning objectives, scenarios, key concepts
- **🎯 Real-World Scenarios** demonstrating practical applications
- **✏️ Hands-On Exercises** with step-by-step guidance
- **💡 Hints & Solutions** for self-paced learning
- **🔍 Reflection Prompts** to deepen understanding
- **📊 Sample Data** for exercises and experiments
- **🌐 HTML Renders** for easy online viewing

---

## 🔧 Technologies & Dependencies

### ML Fundamentals
- **Core:** NumPy, Pandas, Scikit-learn, Matplotlib
- **Advanced:** Scikit-image, SciPy, UMAP, XGBoost

### LangChain Mastery
- **Core:** LangChain, OpenAI (or alternative LLM), embeddings
- **Integrations:** LangSmith, vector stores (FAISS, Chroma)

### LangGraph Workflows
- **Core:** LangGraph, LangChain, state management
- **Tools:** Visualization, persistence layers

### Crew.ai Multi-Agents
- **Core:** Crew.ai, agent orchestration
- **Integrations:** LLM providers, tool libraries

---

## 📝 Features

✅ **Self-Paced Learning** — Start whenever, go at your own speed  
✅ **Progressive Difficulty** — Each week builds on previous knowledge  
✅ **Practical Projects** — Real scenarios from industry  
✅ **Code Examples** — Copy-paste ready, fully commented  
✅ **Multiple Formats** — Jupyter notebooks + HTML renders  
✅ **Searchable Content** — GitHub Pages compatible  
✅ **Free & Open** — No signup required  

---

## 🤝 Contributing

To suggest improvements or report issues:
1. Check existing materials
2. Create a detailed description
3. Submit feedback

---

## 📄 License

Educational materials for learning and reference. Use freely for educational purposes.

---

## 🎯 Next Steps

1. **Start Here:** Open [`index.html`](./index.html) in your browser
2. **Choose a Course:** Click on a course card to explore
3. **Follow the Path:** Week by week, lesson by lesson
4. **Practice:** Work through exercises and assignments
5. **Build:** Create projects using what you've learned

---

**Last Updated:** November 2025  
**Platform:** AI Frameworks Learning Platform  
**Status:** ✅ All courses available and ready to use
