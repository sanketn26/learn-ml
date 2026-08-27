# Learn ML

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20me%20a%20coffee-FFDD00?style=flat-square)](https://buymeacoffee.com/sanketn)

Courses for working software engineers who want to ship ML and LLM systems without a math degree. This is not a beginner programming course.

Lessons are markdown. Exercises are ordinary Python. The site is GitHub Pages (MkDocs). There are no Jupyter notebooks.

## Courses

1. **Applied ML Foundations for SaaS Analytics** (weeks 0–20; 18–20 optional DL)  
   Analogy → visual → math → predict → run → compare → explain. Ends in a real job: `as_of` features, a gate, tonight’s CSV.

2. **LangChain** (7 weeks) — prompts, tools, RAG, a golden file that can fail CI.

3. **LangGraph** (5 weeks) — state machines, resume, don’t charge twice.

4. **CrewAI** (4 weeks) — roles, tickets, crews.

## Read it

[https://sanketn26.github.io/learn-ml/](https://sanketn26.github.io/learn-ml/)

Source is `main`. Actions builds the site. Pages source must be **GitHub Actions** (not a branch). See [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md).

```bash
pip install mkdocs-material
mkdocs serve
```

## Do the exercises

```bash
git clone https://github.com/sanketn26/learn-ml.git
cd learn-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python exercises/ml/week-00/starter.py
```

Each ML week is `exercises/ml/week-XX/starter.py`. Run from the repo root.

Python fighting you, or prefer an isolated environment? `docker build -t learn-ml . && docker run --rm -it -p 8000:8000 learn-ml` gets you a shell with the course, labs, and data already baked in — no `pip install` needed. Or open the repo in VS Code and **Reopen in Container**. See [docs/getting-started.md](docs/getting-started.md#optional-run-it-in-docker-instead).

No GPU. Default loaders sample ~8k customers so a week finishes in a few minutes on 8 GB RAM. The billing table has ~49k customers (observation end 2024-11-30).

## Before you start

You should already be able to write and debug a small program, use a terminal, install dependencies in a virtual environment, read a stack trace, work with Git, and understand basic data ideas such as rows, columns, types, joins, and APIs. Prior Python, ML, calculus, and linear algebra are not required; programming fluency is.

Read the candid [readiness checklist](docs/getting-started.md#this-is-not-beginner-study-material) before committing to the course. If the checklist is unfamiliar, learn those foundations first. Otherwise you will spend your time fighting Python, SQL, Git, and the shell instead of learning ML.

## Repo layout

```
docs/                 # lessons (MkDocs source, this is the course)
exercises/ml/         # starter.py for each week
pipelines/            # train / score / promote
tests/                # pytest gates
eval/                 # golden tickets
lib/course_data.py    # CloudWave loaders
data/                 # synthetic SaaS CSVs / JSON
mkdocs.yml
```

## Datasets

See [docs/data.md](docs/data.md) or [DATASET_GUIDE.md](DATASET_GUIDE.md).
