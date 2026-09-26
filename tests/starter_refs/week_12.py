import importlib.util
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_spec = importlib.util.spec_from_file_location("w12", Path(__file__).resolve().parents[2] / "exercises/ml/week-12/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def task_1():
    return PCA(n_components=2).fit_transform(w.X)


def task_2():
    cum = np.cumsum(PCA().fit(w.X).explained_variance_ratio_)
    k = int(np.argmax(cum >= 0.8)) + 1
    pca = PCA(n_components=k).fit(w.X)
    err = ((w.X - pca.inverse_transform(pca.transform(w.X))) ** 2).sum(axis=1)
    return k, w.sample["user_id"].to_numpy()[np.argsort(-err)[:8]].tolist()
