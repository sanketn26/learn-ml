import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor
from pipelines.split import snapshot_split

train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)


def _fit(**kw):
    return Pipeline([("prep", make_preprocessor()), ("gbt", GradientBoostingClassifier(random_state=42, **kw))]).fit(train[FEATURE_COLS], y_train)


shallow = _fit(n_estimators=40, max_depth=2)


def task_1():
    return pd.Series(shallow.named_steps["gbt"].feature_importances_, index=shallow.named_steps["prep"].get_feature_names_out())


def task_2():
    def pair(m):
        return (roc_auc_score(y_train, m.predict_proba(train[FEATURE_COLS])[:, 1]),
                roc_auc_score(y_test, m.predict_proba(test[FEATURE_COLS])[:, 1]))
    return {"shallow": pair(shallow), "deep": pair(_fit(n_estimators=80, max_depth=8))}
