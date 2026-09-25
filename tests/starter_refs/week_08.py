import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipelines.contract import validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon

as_of = AS_OF_DEFAULT
df = build_features(as_of=as_of, n=None)
frame, y = drop_unlabelled(df, label_churn_in_horizon(df, as_of))
X_train, X_test, y_train, y_test = train_test_split(frame[FEATURE_COLS], y, test_size=0.2, random_state=42, stratify=y)
gbt = Pipeline([("prep", make_preprocessor()),
                ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42))]).fit(X_train, y_train)
proba = gbt.predict_proba(X_test)[:, 1]


def task_1():
    return y.mean(), frame["is_churned"].mean()


def task_2():
    return label_churn_in_horizon(df, as_of, horizon_days=30, observation_end=as_of + pd.Timedelta(days=10))


def task_3():
    return {"roc_auc": roc_auc_score(y_test, proba), "pr_auc": average_precision_score(y_test, proba),
            "dummy_pr_auc": average_precision_score(y_test, np.full(len(y_test), y_train.mean())),
            "precision_at_80": y_test.to_numpy()[np.argsort(-proba)[:80]].mean()}


def task_4():
    row = X_test.iloc[0]
    payload = {k: (str(row[k]) if k == "plan_type" else float(row[k])) for k in FEATURE_COLS}
    try:
        validate({**payload, "churn_date": "2024-07-01"})
    except ValueError as exc:
        return exc


def task_5():
    return calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
