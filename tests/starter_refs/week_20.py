import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location("w20", Path(__file__).resolve().parents[2] / "exercises/ml/week-20/starter.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def _encode(text, n=32):
    ids = [min(ord(c), 126) for c in str(text).lower()[:n]]
    return ids + [0] * (n - len(ids))


ids = np.array([_encode(t) for t in w.fb["feedback_text"]], dtype=np.int64)
idx = np.random.default_rng(0).permutation(len(ids))
cut = int(0.8 * len(ids))
Xtr, Xte, ytr, yte = ids[idx[:cut]], ids[idx[cut:]], w.y[idx[:cut]], w.y[idx[cut:]]


class TinyTransformer(nn.Module):
    def __init__(self, use_pos=True, vocab=127, d=24, ntok=32):
        super().__init__()
        self.use_pos = use_pos
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(ntok, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=48, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d, 1)

    def forward(self, t):
        x = self.emb(t)
        if self.use_pos:
            x = x + self.pos(torch.arange(t.shape[1]).expand_as(t))
        return self.head(self.enc(x).mean(dim=1)).squeeze(-1)


def _fit_eval(use_pos, epochs=4):
    torch.manual_seed(0)
    model = TinyTransformer(use_pos)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(Xtr), 256):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(torch.tensor(Xtr[i:i + 256])), torch.tensor(ytr[i:i + 256], dtype=torch.float32)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float(((model(torch.tensor(Xte)).sigmoid().numpy() > 0.5) == yte).mean())


def task_1():
    return {"with_pos": _fit_eval(True), "without_pos": _fit_eval(False)}


def task_2():
    torch.manual_seed(0)
    tokens = ["login", "failed", "again"]
    E = torch.randn(3, 4)
    Wq, Wk = torch.randn(4, 4), torch.randn(4, 4)
    weights = torch.softmax((E @ Wq) @ (E @ Wk).T / 2, dim=-1)
    return pd.DataFrame(weights.numpy(), index=tokens, columns=tokens)
