# Learn ML

Courses for software engineers who want to ship ML and LLM systems without a math degree.

Lessons are markdown. Exercises are ordinary Python. The site is GitHub Pages (MkDocs). There are no Jupyter notebooks.

## Courses

1. **Applied ML Foundations for SaaS Analytics** (weeks 0–15)  
   Analogies, pictures, then code. Python, NumPy, Pandas, scikit-learn, a little PyTorch.  
   Written for people who already write software.

2. **LangChain** (6 weeks) — prompts, memory, agents, RAG, eval, production.

3. **LangGraph** (4 weeks) — state machines, branches, checkpoints, human-in-the-loop.

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

No GPU. Default loaders sample ~8k customers so a week finishes in a few minutes on 8 GB RAM.

## Repo layout

```
docs/                 # lessons (MkDocs source, this is the course)
exercises/ml/         # starter.py for each week
lib/course_data.py    # CloudWave loaders
data/                 # synthetic SaaS CSVs / JSON
mkdocs.yml
```

## Datasets

See [docs/data.md](docs/data.md) or [DATASET_GUIDE.md](DATASET_GUIDE.md).
