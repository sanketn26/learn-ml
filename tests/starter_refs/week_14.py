import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

_spec = importlib.util.spec_from_file_location("w14", Path(__file__).resolve().parents[2] / "exercises/ml/week-14/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def _auc(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def task_1():
    lr = LogisticRegression(max_iter=1000).fit(w.X_train_t, w.y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="identity", max_iter=50, random_state=42).fit(w.X_train_t, w.y_train)
    return _auc(lr, w.X_test_t, w.y_test), _auc(mlp, w.X_test_t, w.y_test)


def task_2():
    big = MLPClassifier(hidden_layer_sizes=(128, 128, 128), max_iter=30, random_state=42).fit(w.X_train_t, w.y_train)
    return _auc(big, w.X_train_t, w.y_train), _auc(big, w.X_test_t, w.y_test)


def task_4():
    Xt = torch.tensor(np.asarray(w.X_train_t, dtype=np.float32))
    yt = torch.tensor(w.y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

    def run(zero_grad):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(Xt.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        out = []
        for _ in range(5):
            if zero_grad:
                opt.zero_grad()
            loss = nn.BCEWithLogitsLoss()(net(Xt), yt)
            loss.backward()
            opt.step()
            out.append(float(loss))
        return out
    return run(True), run(False)
