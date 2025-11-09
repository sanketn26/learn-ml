# Dataset Access & Download Guide

## 📊 ML Fundamentals Course Datasets

All datasets are included in the repository and available for download in multiple formats.

### Quick Download Links

| Dataset | Records | Size | Format | Download |
|---------|---------|------|--------|----------|
| `subscriptions.csv` | 50,000 | 8 MB | CSV | [Download](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/subscriptions.csv) |
| `user_events.csv` | 220,000 | 15 MB | CSV | [Download](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/user_events.csv) |
| `feature_usage.csv` | 160,000 | 12 MB | CSV | [Download](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feature_usage.csv) |
| `feedback.json` | 10,000 | 3 MB | JSON | [Download](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feedback.json) |
| `product_catalog.csv` | 300 | <1 MB | CSV | [Download](https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/product_catalog.csv) |

### Dataset Descriptions

#### 1. **subscriptions.csv**
**Purpose:** Customer subscription lifecycle and events

**Columns:**
- `customer_id` - Unique customer identifier
- `subscription_date` - Date subscription started
- `status` - Current status (active, churned, upgraded)
- `plan_type` - Subscription plan (free, pro, enterprise)
- `monthly_cost` - Monthly subscription cost
- `upgrade_date` - Date of last upgrade (if applicable)
- `churn_date` - Date of cancellation (if applicable)

**Use Cases:**
- Customer retention analysis
- Churn prediction
- Revenue forecasting
- Upgrade pattern analysis

#### 2. **user_events.csv**
**Purpose:** User activity and interaction telemetry

**Columns:**
- `user_id` - Unique user identifier
- `timestamp` - Event timestamp
- `event_type` - Type of activity (login, click, purchase, etc.)
- `feature_used` - Feature accessed
- `session_duration` - Duration in seconds
- `device_type` - Device used (desktop, mobile, tablet)
- `success` - Whether event was successful (true/false)

**Use Cases:**
- User engagement analysis
- Feature usage patterns
- Conversion funnel analysis
- Session duration analysis

#### 3. **feature_usage.csv**
**Purpose:** Feature adoption and usage metrics

**Columns:**
- `feature_id` - Unique feature identifier
- `user_id` - User using the feature
- `usage_count` - Number of times used
- `last_used` - Date last used
- `adoption_rate` - Percentage adoption
- `satisfaction_score` - User satisfaction rating (1-5)

**Use Cases:**
- Feature adoption analysis
- Feature utilization metrics
- Product performance tracking
- User satisfaction correlation

#### 4. **feedback.json**
**Purpose:** Customer feedback and sentiment

**Fields per entry:**
- `feedback_id` - Unique feedback identifier
- `customer_id` - Customer providing feedback
- `date_submitted` - Date feedback submitted
- `rating` - Overall rating (1-5)
- `category` - Feedback category (bug, feature_request, support, other)
- `text` - Feedback text
- `sentiment` - Sentiment analysis (positive, neutral, negative)

**Use Cases:**
- Sentiment analysis
- Feature request prioritization
- Support ticket categorization
- Customer satisfaction tracking

#### 5. **product_catalog.csv**
**Purpose:** Product and feature definitions

**Columns:**
- `product_id` - Unique product identifier
- `product_name` - Name of product
- `category` - Product category
- `launch_date` - Product launch date
- `status` - Product status (active, deprecated, beta)
- `price_tier` - Pricing tier (free, standard, premium)

**Use Cases:**
- Product portfolio analysis
- Pricing strategy
- Product lifecycle analysis
- Cross-sell opportunities

### Local Access

All datasets are available locally in the `/data/` directory:

```bash
cd learn-ml
ls -lh data/
```

### Using Datasets in Notebooks

Datasets are loaded using relative paths:

```python
import pandas as pd

# Load CSV files
subscriptions = pd.read_csv('../data/subscriptions.csv')
user_events = pd.read_csv('../data/user_events.csv')
feature_usage = pd.read_csv('../data/feature_usage.csv')
product_catalog = pd.read_csv('../data/product_catalog.csv')

# Load JSON file
import json
with open('../data/feedback.json', 'r') as f:
    feedback = json.load(f)
feedback_df = pd.DataFrame(feedback)
```

### Data Characteristics

- **Total Records:** 440,000+
- **Total Size:** 29 MB
- **Synthetic Data:** All datasets are synthetically generated
- **Realistic Patterns:** Generated with real-world SaaS business patterns
- **Time Period:** Covers 1 year of simulated activity
- **Privacy:** No real customer data included

### Git Large File Storage (LFS)

If repository size becomes an issue, datasets can be managed with Git LFS:

```bash
# Install Git LFS
git lfs install

# Track CSV files with LFS
git lfs track "data/*.csv"
git lfs track "data/*.json"

# Commit changes
git add .gitattributes data/
git commit -m "Enable Git LFS for datasets"
```

### External Storage Options

For larger implementations, consider:

1. **AWS S3**
   - Host datasets on S3
   - Link to raw data in notebooks
   - Cost-effective for large volumes

2. **Azure Blob Storage**
   - Similar to S3
   - Good for enterprise deployments

3. **Google Cloud Storage**
   - Alternative cloud option
   - GCP integration with notebooks

4. **Data Package Services**
   - Zenodo (research data)
   - Figshare (data sharing)
   - Open Science Framework

### Downloading All Data at Once

```bash
# Clone entire repository with data
git clone https://github.com/sanketn26/learn-ml.git
cd learn-ml

# Or download just the data using a script
curl -O https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/subscriptions.csv
curl -O https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/user_events.csv
curl -O https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feature_usage.csv
curl -O https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/feedback.json
curl -O https://raw.githubusercontent.com/sanketn26/learn-ml/main/data/product_catalog.csv
```

### Data Verification

Verify dataset integrity using MD5 checksums:

```bash
# Generate checksums
md5sum data/*.csv data/*.json > checksums.md5

# Verify checksums
md5sum -c checksums.md5
```

### Version Control

Datasets are versioned with the code repository. For dataset updates:

```bash
# Check dataset version history
git log --oneline data/

# View changes in specific dataset
git show commit_hash:data/subscriptions.csv
```

### FAQ

**Q: Can I use these datasets for my own projects?**
A: Yes! Datasets are synthetic and available for educational use.

**Q: How often are datasets updated?**
A: Datasets are stable for this course. Updates noted in course release notes.

**Q: What if I want larger datasets?**
A: Use a data generator (Faker library in Python) to create more records following the same schema.

**Q: Can I contribute my own datasets?**
A: Yes! Open an issue or PR on GitHub to propose additions.

---

**Last Updated:** November 2025

For questions, open an issue on GitHub: https://github.com/sanketn26/learn-ml/issues
