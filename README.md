# 🎓 AI Frameworks Learning Platform

A comprehensive multi-course learning platform covering machine learning, LLM orchestration, workflow automation, and multi-agent systems.

## 📚 Courses Included

1. **Applied ML Foundations for SaaS Analytics** (weeks 0–15)
   - Written for software engineers — no calculus or stats degree required
   - Analogies, pictures, and “how to think about it” before any formula
   - **Python**, NumPy, Pandas, **PyTorch**
   - ML fundamentals: regression, classification, overfitting, bias, variance
   - Deep learning: training loop, CNNs, RNNs, Transformers
   - Notebooks: `notebooks/week-00-saas.ipynb` through `week-15-saas.ipynb`

2. **LangChain Fundamentals** (6 weeks)
   - LLM chains, prompts, memory, and agents
   - Production-ready patterns

3. **LangGraph Workflows** (4 weeks)
   - Advanced graph-based workflow automation
   - Multi-step reasoning patterns

4. **Crew.ai Multi-Agent Systems** (4 weeks)
   - Multi-agent orchestration and collaboration
   - Complex task automation

## 📊 Datasets

Located in `data/` (synthetic, production-like patterns):

- `subscriptions.csv` (50K rows) - Customer subscriptions
- `user_events.csv` (220K rows) - User behavior tracking
- `feature_usage.csv` (160K rows) - Feature adoption metrics
- `feedback.json` (10K rows) - Customer feedback
- `product_catalog.csv` (300 rows) - Product information

**Quick access:** See `DATASET_GUIDE.md` for download links and schema details.

## 🚀 Getting Started

### View Content Online
- Open `/docs/` folder for rendered HTML versions
- All 26 lessons available with GitHub links and navigation

### Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sanketn26/learn-ml.git
   cd learn-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook notebooks/
   ```

   ML notebooks are sized for a normal laptop: **no GPU**, about **8 GB RAM**, and each week should finish in a few minutes on CPU. Modeling weeks sample ~8,000 customers. Install `torch` only when you reach Week 11+.

5. **Start with week 1:**
   - ML Fundamentals: `week-01-saas.ipynb`
   - Or choose another course path

## 🔍 Repository Structure

```
learn-ml/
├── notebooks/              # 26 Jupyter notebooks (all courses)
├── docs/                   # HTML-rendered versions of all lessons
├── data/                   # 5 datasets (CSV/JSON format)
├── solutions/              # Solution notebooks (optional)
├── enhance_html.py         # Automation script for HTML generation
├── sitemap.xml             # SEO configuration for search engines
├── robots.txt              # Search engine directives
├── README.md               # This file
├── DATASET_GUIDE.md        # Data access and documentation
└── GITHUB_ACTIONS_SETUP.md # CI/CD workflow guide
```

## 🎯 Course Navigation

Each course includes 4–16 weeks of content. Navigate using:
- **Previous/Next buttons** in HTML versions (`docs/`)
- **GitHub links** on each page to access notebooks directly
- **Setup instructions** on every page for local execution

## 🛠 GitHub Actions Setup

For automated HTML generation and GitHub Pages deployment, see `GITHUB_ACTIONS_SETUP.md`.

## 📝 Feature Highlights

✅ **26 structured lessons** covering 4 different frameworks  
✅ **Interactive notebooks** with explanations, code, exercises  
✅ **Realistic datasets** with SaaS metrics  
✅ **HTML renderings** for easy online browsing  
✅ **GitHub integration** for direct notebook access  
✅ **Mobile responsive** - works on any device  
✅ **SEO optimized** - searchable and discoverable  
✅ **CI/CD ready** - automate your own publishing  

## 🚀 Deployment

The platform is ready for GitHub Pages deployment:

1. Push to GitHub
2. Enable GitHub Pages in repository settings
3. Select `main` branch as source
4. Visit your site at `https://username.github.io/learn-ml/`

For automated HTML rendering, see `GITHUB_ACTIONS_SETUP.md`.

## 📚 Additional Resources

- **Dataset Guide:** `DATASET_GUIDE.md` - Data access, schemas, and examples
- **CI/CD Setup:** `GITHUB_ACTIONS_SETUP.md` - GitHub Actions workflow configuration
- **HTML Content:** `/docs/` - All 26 lessons rendered as HTML

## 📧 Questions?

Each notebook includes:
- Scenario-driven explanations
- Code cells with outputs
- Practice exercises
- Reflection prompts

Work through the material at your own pace!
