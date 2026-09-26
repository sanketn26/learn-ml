import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

_spec = importlib.util.spec_from_file_location("w18", Path(__file__).resolve().parents[2] / "exercises/ml/week-18/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


class UsageCNN(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.pool(F.relu(self.conv(x.unsqueeze(1)))).squeeze(-1)).squeeze(-1)


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(12, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def _fit_eval(make, epochs=40):
    torch.manual_seed(0)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb, yb = torch.tensor(w.Xtr), torch.tensor(w.ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(xb), yb).backward()
        opt.step()
    with torch.no_grad():
        p = model(torch.tensor(w.Xte)).sigmoid().numpy()
    return float(average_precision_score(w.yte, p)), float(roc_auc_score(w.yte, p))


def task_1():
    return {k: _fit_eval(lambda k=k: UsageCNN(k)) for k in (3, 5)}


def task_2():
    return _fit_eval(Dense)
