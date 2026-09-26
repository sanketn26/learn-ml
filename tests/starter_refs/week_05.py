import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from lib.course_data import find_data_dir

DATA = find_data_dir()
subs = pd.read_csv(DATA / "subscriptions.csv")


def task_1():
    paid = subs[subs["plan_type"] != "free"]
    table = pd.crosstab(paid["plan_type"], paid["is_churned"])
    return table, chi2_contingency(table)[1]


def task_2():
    fb = pd.read_json(DATA / "feedback.json", lines=True)
    bug = fb.loc[fb["category"] == "bug", "sentiment_score"]
    praise = fb.loc[fb["category"] == "praise", "sentiment_score"]
    return bug.mean(), praise.mean(), ttest_ind(bug, praise, equal_var=False).pvalue


def task_3():
    rng = np.random.default_rng(0)
    out = {}
    for n in (100, 400, 1000):
        hits = 0
        for _ in range(300):
            a, b = rng.binomial(1, 0.16, n), rng.binomial(1, 0.20, n)
            hits += chi2_contingency([[a.sum(), n - a.sum()], [b.sum(), n - b.sum()]])[1] < 0.05
        out[n] = hits / 300
    return out
