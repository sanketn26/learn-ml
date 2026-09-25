import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines.features import FEATURE_COLS, NUMERIC, build_features, make_preprocessor

df = build_features(as_of="2024-06-01")
X, y = df[FEATURE_COLS], df["is_churned"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
forest = Pipeline([("prep", make_preprocessor()),
                   ("model", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2))]).fit(X_train, y_train)
proba = forest.predict_proba(X_test)[:, 1]
y_np = y_test.to_numpy()


def task_1():
    return proba, y_np, y_np[np.argsort(-proba)[:100]].mean()


def task_2():
    rows = []
    for cut in np.arange(0.1, 1.0, 0.1):
        pred = proba >= cut
        rows.append({"cut": cut, "flagged": int(pred.sum()),
                     "precision": precision_score(y_np, pred, zero_division=0), "recall": recall_score(y_np, pred, zero_division=0)})
    return pd.DataFrame(rows)


def task_3():
    keep = [c for c in NUMERIC if c != "tenure_so_far"]
    prep = ColumnTransformer([("num", StandardScaler(), keep), ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"])])
    ablated = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2))])
    ablated.fit(X_train[keep + ["plan_type"]], y_train)
    return roc_auc_score(y_np, proba), roc_auc_score(y_np, ablated.predict_proba(X_test[keep + ["plan_type"]])[:, 1])
