# Getting started

You do not need Jupyter. You do not need a GPU.

## This is not beginner study material

This course is beginner-friendly **about machine learning**, not beginner-friendly **about software engineering**. It teaches ML ideas by comparing them with functions, APIs, SQL joins, tests, CI, batch jobs, state machines, and on-call incidents. Those comparisons only help if the software concepts are already familiar.

It is also not a complete reference or exam-preparation text. The lessons deliberately trade formal proofs and exhaustive theory for engineering intuition, runnable examples, failure modes, and shipping decisions. If you need a first programming course, a mathematical ML textbook, or a framework API reference, use one before or alongside this course.

### What you must already have

Before Week 0, you should be able to:

- write a small program in some language using variables, functions, collections, conditionals, loops, and classes;
- debug from an error message and stack trace instead of only copying a replacement snippet;
- use a terminal, navigate directories, create a virtual environment, and install dependencies;
- use Git well enough to clone a repository, inspect a diff, and preserve your work;
- read tabular data as rows, columns, types, missing values, and a declared grain;
- understand the purpose of SQL `SELECT`, `GROUP BY`, and `JOIN`, even if the exact syntax needs refreshing;
- recognize an API request/response, a schema or contract, a unit test, and a batch job; and
- tolerate light algebra such as averages, percentages, ratios, and reading a formula one symbol at a time.

For the LangChain, LangGraph, and CrewAI tracks, first complete the relevant ML material or bring equivalent experience. You should also understand HTTP APIs, JSON, environment variables, retries, persistence, and the fact that LLM output is untrusted input. Some exercises require a paid model API key and can incur usage charges.

### A five-minute readiness check

You are ready if you can create a script that reads a CSV, groups rows by a key, prints a result, adds one assertion, and commits the change—and can make progress when the first run fails. Looking up syntax is normal.

If every noun in that sentence is new, stop here and take introductory Python, terminal/Git, and SQL courses first. Skipping that preparation will not make this course faster. You will spend the course fighting the tools while the ML reasoning passes by.

You do **not** need prior ML, calculus, linear algebra, a statistics degree, Jupyter, or a GPU.

## Read the lessons

This site *is* the course. Open a week, read the analogy, look at the picture, then the code block.

If you only want the intuition, stop there. The code is proof, not a ritual.

## Run the exercises

```bash
git clone https://github.com/sanketn26/learn-ml.git
cd learn-ml
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python exercises/ml/week-00/starter.py
pytest tests/                 # from week 16 on
```

Each ML week has:

```
exercises/ml/week-00/
  README.md     # the tasks
  starter.py    # run this, fill in the TODOs
```

The same tasks are also on the site under **ML Fundamentals → Exercises**.

Work from the repo root so `lib/course_data.py` can find `data/`.

## What to install

**ML weeks 0–10**

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

**ML weeks 14, 18–20** (CPU is enough)

```bash
pip install torch
```

**LangChain / LangGraph / CrewAI** — only when you reach those courses. Read the [framework track guide](framework-tracks.md) first. Use a separate environment because these libraries evolve independently:

```bash
python3.11 -m venv .venv-framework
source .venv-framework/bin/activate
pip install -r requirements-frameworks.txt
```

CrewAI is optional and heavier. Install its isolated environment only when you begin that track:

```bash
python3.11 -m venv .venv-crewai
source .venv-crewai/bin/activate
pip install -r requirements-crewai.txt
```

Concept demos use ordinary Python or fake models and do not need an API key. Integration demos do:

```
OPENAI_API_KEY=sk-...
```

## Preview this site locally

```bash
pip install mkdocs-material
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Laptop budget

No GPU. Models use a sample of ~8,000 customers (sequences: ~3,000 users) so each week finishes in a few minutes on an 8 GB machine. Pass `n=None` to `load_customer_360` only if you want the full 50k.
