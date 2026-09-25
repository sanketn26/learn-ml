import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("w11", Path(__file__).resolve().parents[2] / "exercises/ml/week-11/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
y, scores, test, pk = w.y_test, w.scores, w.test, w.precision_at_k


def task_1():
    rng = np.random.default_rng(0)
    rankers = {"gbt": scores, "n_support": test["n_support"].to_numpy(),
               "-log_usage": -test["log_usage"].to_numpy(), "random": rng.random(len(test))}
    return {n: (pk(y, s), float(y[np.argsort(-s)[:80]].sum() / y.sum())) for n, s in rankers.items()}


def task_2():
    return {k: pk(y, scores, k) for k in (20, 80, 200)}


def task_5():
    low = -test["log_usage"].to_numpy()
    rng = np.random.default_rng(0)
    mp, gap = [], []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        a = pk(y[i], scores[i])
        mp.append(a)
        gap.append(a - pk(y[i], low[i]))
    return (*np.percentile(mp, [2.5, 97.5]), *np.percentile(gap, [2.5, 97.5]))
