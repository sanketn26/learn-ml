import importlib.util
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_spec = importlib.util.spec_from_file_location("w9", Path(__file__).resolve().parents[2] / "exercises/ml/week-09/starter.py")
w9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w9)
train, test, y_train, y_test = w9.train, w9.test, w9.y_train, w9.y_test
num = ["mrr", "tenure_so_far", "log_usage", "features_adopted", "total_events", "n_support"]


def _forest():
    prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"])])
    return Pipeline([("prep", prep), ("m", RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42, n_jobs=2))])


cols = num + ["plan_type"]
raw_pred = _forest().fit(train[cols], y_train).predict(test[cols])
log_pred = np.expm1(_forest().fit(train[cols], np.log1p(y_train)).predict(test[cols]))


def task_1():
    return mean_absolute_error(y_test, raw_pred), mean_absolute_error(y_test, log_pred), log_pred


def task_2():
    frame = test[["plan_type"]].assign(abs_err=(y_test.to_numpy() - log_pred).__abs__())
    return frame.groupby("plan_type").agg(mae=("abs_err", "mean"))


def task_3():
    fake = lambda f: f["mrr"] * f["tenure_so_far"] / 30  # noqa: E731
    lin = LinearRegression().fit(train[["mrr", "tenure_so_far"]], fake(train))
    return r2_score(fake(test), lin.predict(test[["mrr", "tenure_so_far"]]))
