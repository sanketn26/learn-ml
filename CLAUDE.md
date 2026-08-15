# CLAUDE.md

Guidance for working in this repository.

## What this is

An AI-frameworks learning platform. **Markdown lessons + separate Python exercises**, served with MkDocs on GitHub Pages. No Jupyter.

- **ML Fundamentals** (weeks 0–15): for software engineers without a math background. Each week is analogy → picture → code → “watch out” → “ship / don’t ship,” plus an “If you already write software” mapping.
- **Laptop budget:** no GPU. `load_customer_360` samples ~8k rows; sequence weeks use ~3k users. Override with `n=None` only for the full 50k.
- **LangChain** (6 weeks), **LangGraph** (4), **CrewAI** (4).

Synthetic SaaS data in `data/`. CloudWave is the through-line.

## Layout

```
docs/                 MkDocs source = the course
  ml/week-XX.md       lessons
  ml/exercises/       exercise pages
  langchain/ langgraph/ crewai/
exercises/ml/week-XX/ starter.py the learner runs
lib/course_data.py    loaders (no IPython)
data/                 CSVs / JSON
mkdocs.yml
```

## Commands

```bash
pip install -r requirements.txt
mkdocs serve                          # http://127.0.0.1:8000
python exercises/ml/week-00/starter.py
mkdocs build --strict
```

CI: `.github/workflows/pages.yml` builds MkDocs and deploys GitHub Pages.

## Pedagogy (do not drop)

- Start with a software analogy (SQL, API contract, code review, flaky test).
- Then a picture (ASCII is fine).
- Then a small code block that *proves* the picture.
- Then a foot-gun and a ship / don’t-ship rule.
- Formulas only as “math, translated.”
- Exercises live **next to** the lesson, not inside it. `starter.py` is a file you run in a terminal.

Callout CSS lives in `docs/stylesheets/extra.css`. Admonition types: `think`, `engineer`, `warning`, `success`, `math`, `tip`.

## Adding a week

1. Write `docs/<course>/week-XX.md`.
2. If it is ML, add `docs/ml/exercises/week-XX.md` and `exercises/ml/week-XX/starter.py`.
3. Add the page to `mkdocs.yml` `nav`.
4. Preview with `mkdocs serve`.

## Datasets

`data/subscriptions.csv` (50k), `user_events.csv` (220k), `feature_usage.csv` (160k), `feedback.json` (10k JSON Lines), `product_catalog.csv` (300). Load via `lib.course_data.find_data_dir` / `load_customer_360`. See `DATASET_GUIDE.md`.
