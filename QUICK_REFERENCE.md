# 🎓 Applied ML for SaaS Analytics — Quick Reference Guide

## 📍 Key Files Overview

### Entry Point
- **`index.html`** — Start here! Professional landing page with course overview

### Course Content (12 Weeks)
- **`notebooks/`** — Full Jupyter notebooks with interactive content
  - `week-01-saas.ipynb` — NumPy Fundamentals
  - `week-02-saas.ipynb` — Pandas ETL  
  - `week-03-saas.ipynb` — Data Visualization
  - `week-04-saas.ipynb` — Statistical Analysis
  - `week-05-saas.ipynb` — Feature Engineering
  - `week-06-saas.ipynb` — Classification
  - `week-07-saas.ipynb` — Regression
  - `week-08-saas.ipynb` — Clustering
  - `week-09-saas.ipynb` — Dimensionality Reduction
  - `week-10-saas.ipynb` — Ensemble Methods
  - `week-11-saas.ipynb` — Deep Learning
  - `week-12-saas.ipynb` — Capstone Project

### HTML Renders (For GitHub Pages)
- **`docs/`** — Pre-rendered HTML files for online viewing
  - Same 12 weeks available as `.html` files

### Sample Data
- **`data/subscriptions.csv`** — 50K customer subscriptions
- **`data/user_events.csv`** — 220K user activity events
- **`data/feature_usage.csv`** — 160K feature engagement records
- **`data/feedback.json`** — 10K customer feedback entries
- **`data/product_catalog.csv`** — 300 product/feature definitions

### Documentation
- **`README.md`** — Original course overview
- **`ENHANCEMENTS_SUMMARY.md`** — Complete enhancement details (342 lines)
- **`COMPLETION_CHECKLIST.md`** — Validation checklist
- **`QUICK_REFERENCE.md`** — This file

### Assignments (Optional)
- **`assignments/`** — Weekly assignment prompts
  - `week-01-assignment.md` through `week-12-assignment.md`

### Solutions (Reference)
- **`solutions/README.md`** — Solution guidelines

---

## 🚀 Getting Started

### Option 1: Local Jupyter (Recommended for Learning)
```bash
cd /workspaces/learn-ml
jupyter notebook notebooks/
# Opens at http://localhost:8888
# Click week-01-saas.ipynb to start
```

### Option 2: View HTML Online
1. Push to GitHub
2. Enable Pages: Settings → Pages → Source: `/docs`
3. Access at `https://youruser.github.io/learn-ml/`

### Option 3: Open index.html Locally
```bash
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or just double-click in Windows
```

---

## 📚 Course Structure

### Learning Path (12 Weeks)

**Foundation (Weeks 1-2)**
- Week 1: NumPy arrays, vectorization, broadcasting
- Week 2: Pandas DataFrames, joins, aggregation

**Exploration (Weeks 3-4)**
- Week 3: Visualization for insights
- Week 4: Statistical testing for validation

**Preparation (Week 5)**
- Week 5: Feature engineering, preprocessing

**Modeling (Weeks 6-10)**
- Week 6: Classification (predict churn)
- Week 7: Regression (predict value)
- Week 8: Clustering (discover segments)
- Week 9: Dimensionality reduction
- Week 10: Ensemble methods

**Advanced (Weeks 11-12)**
- Week 11: Deep learning introduction
- Week 12: Capstone project

---

## ✅ What's Included in Each Week

Each notebook contains:

1. **Learning Objectives** — 5-7 specific goals
2. **Real-World Scenario** — SaaS-specific context
3. **Key Concepts** — Theory + examples
4. **Hands-on Exercises** — 3-6 concrete problems
5. **Hints** — Guidance (collapsible)
6. **Solutions** — Working code (collapsible)
7. **Code Demo** — Executable example
8. **Reflection Questions** — Critical thinking
9. **Practice Assignment** — Synthesis task
10. **Next Steps** — Progression to Week N+1

---

## 🎯 Key Topics by Week

| Week | Topic | Key Skills |
|------|-------|-----------|
| 1 | NumPy | Arrays, vectorization, broadcasting |
| 2 | Pandas | DataFrames, joins, aggregation |
| 3 | Visualization | Charts, dashboards, storytelling |
| 4 | Statistics | Hypothesis testing, CI, significance |
| 5 | Features | Encoding, scaling, domain features |
| 6 | Classification | Logistic Regression, Trees, Forests |
| 7 | Regression | Linear, Ridge, RF Regressor |
| 8 | Clustering | K-Means, hierarchical, DBSCAN |
| 9 | PCA | Dimensionality reduction, visualization |
| 10 | Ensembles | Boosting, stacking, optimization |
| 11 | Deep Learning | Neural networks, TensorFlow/Keras |
| 12 | Capstone | Full pipeline, deployment, ethics |

---

## 💾 Sample Datasets

All datasets are synthetically generated with realistic patterns:

### subscriptions.csv
```
Columns: user_id, signup_date, churn_date, plan_tier, region
Size: 50,000 records, 2.0 MB
Use: Cohort analysis, churn prediction, retention curves
```

### user_events.csv
```
Columns: user_id, timestamp, event_type, event_value, region
Size: 220,000 records, 19 MB
Use: Time-series analysis, DAU/MAU, engagement trends
```

### feature_usage.csv
```
Columns: user_id, feature_name, date, usage_count
Size: 160,000 records, 5.9 MB
Use: Feature adoption, usage patterns, correlation with churn
```

### feedback.json
```
Structure: Array of {user_id, timestamp, category, sentiment, text}
Size: 10,000 records, 1.4 MB
Use: Sentiment analysis, NLP (weeks 11-12)
```

### product_catalog.csv
```
Columns: feature_id, feature_name, category, launch_date
Size: 300 records, 14 KB
Use: Joining with usage data for richer context
```

---

## 📊 Exercises Summary

- **Total Exercises:** 24+
- **Total Solutions:** 6+ complete examples
- **Total Code Demos:** 6 executable programs
- **Total Learning Objectives:** 70+
- **Total Reflection Questions:** 24+

### Exercise Types
- **NumPy:** Array operations, broadcasting, aggregations (Week 1)
- **Pandas:** Data cleaning, joins, groupby (Week 2)
- **Visualization:** Charts, dashboards, storytelling (Week 3)
- **Statistics:** Hypothesis tests, confidence intervals (Week 4)
- **Features:** Engineering, scaling, encoding (Week 5)
- **Models:** Classification, regression, clustering (Weeks 6-8)

---

## 🔍 How to Use This Course

### For Self-Paced Learning
1. Start with index.html for context
2. Work through weeks sequentially
3. Execute code cells as you read
4. Try exercises before checking solutions
5. Complete practice assignments

### For Instructor-Led Cohorts
1. Use week sequence for lesson plan
2. Adjust pace based on group
3. Assign practice assignments with deadlines
4. Use reflection questions for discussions
5. Have students build capstone project

### For Reference
- Each week stands somewhat alone
- Use search/find (Ctrl+F) to locate topics
- Solutions section for quick help
- Code demos show best practices

---

## 📝 What Students Will Learn

### Data Processing
- ✓ NumPy vectorization (50-100x faster than Python loops)
- ✓ Pandas groupby/aggregation patterns
- ✓ SQL-like joins in Pandas
- ✓ ETL pipeline design

### Analysis & Insight
- ✓ Statistical significance testing
- ✓ Cohort analysis
- ✓ Retention curves
- ✓ Churn prediction signals

### Modeling
- ✓ Classification (predict binary outcomes)
- ✓ Regression (predict continuous values)
- ✓ Clustering (discover patterns)
- ✓ Ensemble methods (combine models)

### Production
- ✓ Feature engineering best practices
- ✓ Model evaluation metrics
- ✓ Avoiding data leakage
- ✓ Scaling for deployment

### Business
- ✓ SaaS metrics (ARR, MRR, churn, CAC, LTV)
- ✓ Cohort thinking
- ✓ How ML impacts business decisions
- ✓ Cost-benefit of interventions

---

## 🏆 Projects & Capstones

### Week 6 Deliverable
Build a churn prediction model:
- Train on historical subscriptions + features
- Generate risk scores for all customers
- Identify intervention opportunities
- Estimate impact of retention efforts

### Week 7 Deliverable
Build a CLV prediction model:
- Predict customer lifetime value
- Segment by predicted value
- Recommend sales prioritization

### Week 8 Deliverable
Discover customer segments:
- Cluster based on engagement metrics
- Create customer personas
- Recommend targeted strategies

### Week 12 Capstone
Full ML pipeline:
- ETL: Load and prepare data
- Features: Engineer meaningful signals
- Model: Train classification/regression
- Evaluate: Validate performance
- Document: Explain decisions & results
- Deploy: Create reproducible pipeline

---

## 🎓 Prerequisites

### Required Knowledge
- Python basics (variables, loops, functions)
- Statistics basics (mean, std, correlation)
- No ML experience required!

### Required Tools
- Python 3.8+
- Jupyter Notebook or JupyterLab
- NumPy, Pandas, Matplotlib, Scikit-learn, SciPy
- Optional: Plotly for interactive plots, Keras for deep learning

### Installation
```bash
pip install jupyter numpy pandas matplotlib scikit-learn scipy
# Optional:
pip install plotly tensorflow keras
```

---

## 📚 Additional Resources

### NumPy
- Official docs: https://numpy.org/
- Broadcasting guide: https://numpy.org/doc/stable/user/basics.broadcasting.html

### Pandas
- Official docs: https://pandas.pydata.org/
- Cheat sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

### Scikit-learn
- Official docs: https://scikit-learn.org/
- Model selection: https://scikit-learn.org/stable/model_selection.html

### Statistics
- StatQuest videos: https://www.youtube.com/@statquest
- Think Stats book: https://greenteapress.com/thinkstats/

### SaaS Metrics
- "Traction" by Gabriel Weinberg
- "Inspired" by Marty Cagan
- Industry blogs: First Round Review, A16Z

---

## 🤝 Community & Support

### Getting Help
1. **Check solutions** — Collapsible in each week
2. **Review hints** — Guidance for stuck points
3. **Run code** — Execute and inspect output
4. **Ask peers** — Discuss in cohort
5. **Search online** — Stack Overflow, documentation

### Contributing
- Found a bug? File an issue
- Improvement ideas? Submit PR
- Better explanation? We accept contributions!

---

## ✨ Quick Tips

### For Best Learning
- [ ] Code along, don't copy-paste
- [ ] Run exercises before checking solutions
- [ ] Modify code and experiment
- [ ] Use real data (provided datasets)
- [ ] Build your own projects

### For Teaching Others
- [ ] Emphasize SaaS context
- [ ] Use their actual metrics/data
- [ ] Focus on business insights first, math second
- [ ] Encourage lots of exploration
- [ ] Celebrate questions, not just answers

---

## 📞 Questions?

See:
- `ENHANCEMENTS_SUMMARY.md` — Detailed information
- `COMPLETION_CHECKLIST.md` — Validation details
- `README.md` — Course overview
- Each week's "Reflection" section — Think deeper

---

**Last Updated:** November 9, 2025  
**Version:** 1.0 Enhanced  
**Status:** Production Ready ✅
