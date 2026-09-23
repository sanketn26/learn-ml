"""Train, score, and compare — the same way for every encoder.

`fit_predict` trains any model with the signature
`model(tokens, recency, mask, static) -> logits` on the training snapshot
and scores the full test snapshot. `bakeoff` runs each encoder over several
seeds next to the Week-13 GBT, because one seed's AUC is an anecdote.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from capstone_sequence.data import Split
from pipelines.features import FEATURE_COLS, make_preprocessor

ModelFactory = Callable[[int], nn.Module]  # n_static -> model
BUDGET = 80


def _tensors(split: Split, rows=slice(None)):
    return (torch.from_numpy(split.tokens[rows]), torch.from_numpy(split.recency[rows]),
            torch.from_numpy(split.mask[rows]), torch.from_numpy(split.static[rows]))


def fit_predict(make_model: ModelFactory, train: Split, test: Split, epochs: int = 20,
                lr: float = 3e-3, batch: int = 512, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(train.static.shape[1])
    pos = float(train.y.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train.y) - pos) / max(pos, 1.0)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = torch.from_numpy(train.y)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train.y))
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            opt.zero_grad()
            loss_fn(model(*_tensors(train, rows)), y[rows]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*_tensors(test))).numpy()


def gbt_scores(train: Split, test: Split, seed: int = 0) -> np.ndarray:
    """The Week-13 GBT on the aggregated row — the bar every encoder has to clear."""
    gbt = Pipeline([("prep", make_preprocessor()),
                    ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=seed))])
    gbt.fit(train.frame[FEATURE_COLS], train.y.astype(int))
    return gbt.predict_proba(test.frame[FEATURE_COLS])[:, 1]


def metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    top = np.argsort(-scores)[:BUDGET]
    return {
        "auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "dummy_pr_auc": float(y.mean()),
        "hits_at_80": int(y[top].sum()),
    }


def bakeoff(encoders: dict[str, ModelFactory], train: Split, test: Split,
            seeds: tuple[int, ...] = (0, 1, 2), epochs: int = 20) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.append({"model": "gbt (week 13)", "seed": seed, **metrics(gbt_scores(train, test, seed), test.y)})
        for name, make_model in encoders.items():
            rows.append({"model": name, "seed": seed,
                         **metrics(fit_predict(make_model, train, test, epochs=epochs, seed=seed), test.y)})
    table = pd.DataFrame(rows)
    summary = table.groupby("model", sort=False).agg(
        auc=("auc", "mean"), auc_sd=("auc", "std"), pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
        hits_at_80=("hits_at_80", "mean"),
    )
    summary["pr_lift"] = summary["pr_auc"] / float(test.y.mean())
    return summary.round(4)
