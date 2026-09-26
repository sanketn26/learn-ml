import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location("w19", Path(__file__).resolve().parents[2] / "exercises/ml/week-19/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


class SequenceNet(nn.Module):
    def __init__(self, pool="last", hidden=16):
        super().__init__()
        self.pool = pool
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1, :] if self.pool == "last" else out.mean(dim=1)).squeeze(-1)


def _fit_eval(pool, train_x, test_x, epochs=30):
    torch.manual_seed(0)
    model = SequenceNet(pool)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb, yb = torch.tensor(train_x.copy()), torch.tensor(w.ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(xb), yb).backward()
        opt.step()
    with torch.no_grad():
        return float(roc_auc_score(w.yte, model(torch.tensor(test_x.copy())).sigmoid().numpy()))


def task_1():
    return {p: _fit_eval(p, w.Xtr, w.Xte) for p in ("last", "mean")}


def task_2():
    return _fit_eval("last", w.Xtr, w.Xte), _fit_eval("last", w.Xtr[:, ::-1], w.Xte[:, ::-1])
