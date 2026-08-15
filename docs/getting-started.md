# Getting started

You do not need Jupyter. You do not need a GPU.

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

**ML weeks 11, 13–15** (CPU is enough)

```bash
pip install torch
```

**LangChain / LangGraph / CrewAI** — only when you reach those courses, and only if you want to run the snippets. They need an API key in `.env`:

```bash
pip install langchain langgraph python-dotenv
# CrewAI is optional and heavy; install when you start that course
```

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
