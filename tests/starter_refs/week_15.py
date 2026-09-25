import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from pipelines.contract import predict  # noqa: E402
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor  # noqa: E402
from pipelines.labels import drop_unlabelled, label_churn_in_horizon  # noqa: E402
from pipelines.split import snapshot_split  # noqa: E402

train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)


def gbt():
    return Pipeline([("prep", make_preprocessor()), ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42))])


def _auc(tr, ytr, te, yte):
    return roc_auc_score(yte, gbt().fit(tr[FEATURE_COLS], ytr).predict_proba(te[FEATURE_COLS])[:, 1])


model = gbt().fit(train[FEATURE_COLS], y_train)
scores = model.predict_proba(test[FEATURE_COLS])[:, 1]


def task_1():
    snap = build_features(as_of=AS_OF_DEFAULT, n=None)
    snap, y = drop_unlabelled(snap, label_churn_in_horizon(snap, AS_OF_DEFAULT, horizon_days=30))
    a_tr, a_te, ya_tr, ya_te = train_test_split(snap, y, test_size=0.2, random_state=42, stratify=y)
    cut = snap["signup_date"].quantile(0.80)
    b_tr, b_te = snap[snap["signup_date"] <= cut], snap[snap["signup_date"] > cut]
    return {"shuffled": _auc(a_tr, ya_tr, a_te, ya_te),
            "signup_cut": _auc(b_tr, y.loc[b_tr.index], b_te, y.loc[b_te.index]),
            "backtest": roc_auc_score(y_test, scores)}


def task_2():
    artifact = {"pipeline": model, "metrics": {"threshold": 0.5, "model_version": "20240601"}}
    lat = []
    for _, row in test.head(80).iterrows():
        payload = {k: (str(row[k]) if k == "plan_type" else float(row[k])) for k in FEATURE_COLS}
        t0 = time.perf_counter()
        response = predict(payload, artifact)
        lat.append((time.perf_counter() - t0) * 1000)
    return response, float(np.percentile(lat, 95))


def task_3():
    distinct = np.unique(scores)[::-1]
    flagged_at = np.searchsorted(np.sort(-scores), -distinct, side="right")
    cut = float(distinct[flagged_at <= 80][-1])
    return cut, int((scores >= cut).sum())


def task_4():
    fig, axes = plt.subplots(1, 3)
    for ax, col in zip(axes, ["mrr", "log_usage", "tenure_so_far"]):
        ax.hist(train[col], bins=30, alpha=0.5)
        ax.hist(test[col], bins=30, alpha=0.5)
    return fig
