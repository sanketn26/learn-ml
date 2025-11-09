# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI Frameworks Learning Platform containing 26 Jupyter notebooks covering 4 courses:
- **ML Fundamentals** (12 weeks): NumPy, Pandas, Visualization, Scikit-learn, Deep Learning
- **LangChain Fundamentals** (6 weeks): LLM chains, prompts, memory, agents
- **LangGraph Workflows** (4 weeks): Graph-based workflow automation
- **Crew.ai Multi-Agent Systems** (4 weeks): Multi-agent orchestration

All content is educational with synthetic SaaS datasets and HTML-rendered versions for web viewing.

## Repository Structure

```
learn-ml/
├── notebooks/              # ML Fundamentals: week-01-saas.ipynb through week-12-saas.ipynb
├── langchain/
│   ├── notebooks/          # week-01-langchain-basics.ipynb through week-06-*.ipynb
│   ├── docs/              # HTML renders (auto-generated)
│   └── index.html         # LangChain course landing page
├── langgraph/
│   ├── notebooks/          # week-01-graphs-basics.ipynb through week-04-*.ipynb
│   ├── docs/              # HTML renders (auto-generated)
│   └── index.html         # LangGraph course landing page
├── crewai/
│   ├── notebooks/          # week-01-agent-fundamentals.ipynb through week-04-*.ipynb
│   ├── docs/              # HTML renders (auto-generated)
│   └── index.html         # CrewAI course landing page
├── data/                  # Synthetic datasets (CSV/JSON)
├── docs/                  # ML course HTML renders (auto-generated)
├── enhance_html.py        # Adds navigation, links, and styling to HTML files
├── index.html             # Main landing page
└── .github/workflows/     # CI/CD automation
```

## Common Commands

### Running Notebooks Locally

```bash
# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies for ML course
pip install jupyter notebook numpy pandas scikit-learn matplotlib scipy

# Install LangChain dependencies
pip install langchain openai python-dotenv

# Install LangGraph dependencies
pip install langgraph langchain openai

# Install CrewAI dependencies
pip install crewai openai

# Launch Jupyter
jupyter notebook notebooks/  # For ML course
jupyter notebook langchain/notebooks/  # For LangChain
jupyter notebook langgraph/notebooks/  # For LangGraph
jupyter notebook crewai/notebooks/  # For CrewAI
```

### HTML Generation Workflow

This repository uses a two-step process to convert notebooks to web-viewable HTML:

1. **Convert notebooks to HTML:**
```bash
# ML course
jupyter nbconvert --to html --output-dir=docs notebooks/*.ipynb

# LangChain course
cd langchain/notebooks && jupyter nbconvert --to html --output-dir=../docs *.ipynb

# LangGraph course
cd langgraph/notebooks && jupyter nbconvert --to html --output-dir=../docs *.ipynb

# CrewAI course
cd crewai/notebooks && jupyter nbconvert --to html --output-dir=../docs *.ipynb
```

2. **Enhance HTML with navigation and links:**
```bash
python enhance_html.py
```

The `enhance_html.py` script adds:
- Navigation breadcrumbs and prev/next week links
- GitHub links, Colab links, download buttons
- Setup instructions for local execution
- Consistent styling across all pages

### GitHub Actions

The repository includes automated HTML rendering via GitHub Actions:

- **Workflow file:** [.github/workflows/render-notebooks.yml](.github/workflows/render-notebooks.yml)
- **Triggers:** Pushes to main branch
- **Actions:** Converts notebooks to HTML and commits to `docs/`
- **Note:** The current workflow only processes the ML course notebooks. To render all courses, expand the workflow as documented in [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md)

## Architecture Notes

### HTML Enhancement System

The `enhance_html.py` script is the core automation tool that transforms basic Jupyter-rendered HTML into a cohesive learning platform:

**Key Components:**
- **Idempotency check:** Checks if HTML is already enhanced (by looking for `class="notebook-header"`) to prevent duplicate headers
- **Course detection:** Automatically identifies course type from file path (ML, LangChain, LangGraph, CrewAI)
- **Week extraction:** Parses `week-XX` pattern from filenames to determine position in course
- **Navigation generation:** Creates prev/next links based on week number and total course weeks
- **Header injection:** Inserts styled header with breadcrumbs and external links after `<body>` tag
- **Footer injection:** Adds navigation footer before `</body>` tag
- **Course-specific setup:** Customizes pip install commands based on course requirements

**Important:** The script now uses `Path(__file__).parent.absolute()` to automatically detect the repository path, so it works regardless of where the repo is cloned.

**Hard-coded configuration in enhance_html.py:**
- Course week counts: ML=12, LangChain=6, LangGraph=4, CrewAI=4 (lines 18-25)
- GitHub URLs: Points to `sanketn26/learn-ml` repository (lines 41-42)

**To modify enhance_html.py for a fork:**
1. Update GitHub URLs in `create_header_section()` to your repository
2. Adjust week counts in `get_course_info()` if course structure changes

**Idempotency:** The script can be run multiple times safely - it will skip files that have already been enhanced.

### Dataset Architecture

All datasets are stored in `/data/` directory and are synthetic:
- **subscriptions.csv** (50K rows): Customer subscription lifecycle
- **user_events.csv** (220K rows): User activity telemetry
- **feature_usage.csv** (160K rows): Feature adoption metrics
- **feedback.json** (10K rows): Customer feedback with sentiment
- **product_catalog.csv** (300 rows): Product definitions

**Dataset access pattern in notebooks:**
```python
import pandas as pd
subscriptions = pd.read_csv('../data/subscriptions.csv')  # Relative path from notebooks/
```

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for schemas and download links.

### Notebook Naming Convention

All notebooks follow the pattern: `week-XX-{topic}.ipynb`

- **ML course:** `week-01-saas.ipynb` through `week-12-saas.ipynb`
- **LangChain:** `week-01-langchain-basics.ipynb`, `week-02-memory-conversation.ipynb`, etc.
- **LangGraph:** `week-01-graphs-basics.ipynb`, `week-02-workflows.ipynb`, etc.
- **CrewAI:** `week-01-agent-fundamentals.ipynb`, `week-02-task-management.ipynb`, etc.

The week number (01-12) is critical for the HTML enhancement script to generate correct navigation.

## Development Workflows

### Adding a New Notebook

1. Create notebook in appropriate `*/notebooks/` directory following naming convention
2. Run `jupyter nbconvert` to generate HTML
3. Run `python enhance_html.py` to add navigation
4. Update course index.html if adding new week
5. Commit both `.ipynb` and generated `.html` files

### Modifying Existing Notebooks

1. Edit the `.ipynb` file directly or via Jupyter
2. Re-run HTML generation commands (see Common Commands section)
3. Commit changes to both `.ipynb` and `.html`

### Testing HTML Enhancements

```bash
# Generate HTML for a single notebook
jupyter nbconvert --to html notebooks/week-01-saas.ipynb --output-dir=docs

# Run enhancement script
python enhance_html.py

# Open in browser to verify
open docs/week-01-saas.html  # macOS
xdg-open docs/week-01-saas.html  # Linux
```

### GitHub Pages Deployment

The repository is configured for GitHub Pages deployment:

1. Push changes to `main` branch
2. GitHub Actions workflow runs (if enabled)
3. Pages site updates automatically at `https://username.github.io/learn-ml/`

For detailed setup instructions, see [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md).

## Important Files

- **enhance_html.py**: Core automation script - modifies HTML files in-place
- **index.html**: Main landing page with links to all courses
- **sitemap.xml**: SEO sitemap for search engines
- **robots.txt**: Search engine crawling directives
- **DATASET_GUIDE.md**: Complete dataset documentation
- **GITHUB_ACTIONS_SETUP.md**: CI/CD workflow configuration guide

## Notes for AI Assistants

- **Always regenerate HTML** after modifying notebooks using the two-step process
- **If HTML has duplicate headers:** Regenerate HTML files from scratch using `jupyter nbconvert`, then run `python enhance_html.py`
- **The enhancement script is idempotent:** It's safe to run multiple times - already-enhanced files will be skipped
- **Preserve notebook structure**: Notebooks include custom CSS in first cell for styling
- **Respect file patterns**: Week numbers must be zero-padded (01, 02, not 1, 2)
- **Course independence**: Each course (ML/LangChain/LangGraph/CrewAI) operates independently
- **No external dependencies for data**: All datasets are included in the repository
