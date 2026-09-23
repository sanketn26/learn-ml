"""Deep-learning capstone reference solution — three encoders against the GBT.

Run from the repo root:

    python solutions/ml/capstone-sequence/solution.py

CPU, about a minute. Every network sees the event sequence *and* the same
static row the GBT sees, so the only thing a network can add is order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone_sequence.data import MAX_LEN, PAD, VOCAB_SIZE, bakeoff_data
from capstone_sequence.harness import bakeoff

D = 16


class Steps(nn.Module):
    """Event token + recency → one D-dim vector per step (shared by every encoder)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.mix = nn.Linear(D + 1, D)

    def forward(self, tokens, recency):
        return torch.relu(self.mix(torch.cat([self.embed(tokens), recency.unsqueeze(-1)], dim=-1)))


class Head(nn.Module):
    """Pooled sequence (or nothing) + the static row → one logit."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, *parts):
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class StaticOnly(nn.Module):
    """The control: same head, no sequence. If an encoder only matches this, order added nothing."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.head = Head(n_static)

    def forward(self, tokens, recency, mask, static):
        return self.head(static)


class ConvEncoder(nn.Module):
    """Week 18: one detector slid along the events; keep its loudest hit."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.steps = Steps()
        self.conv = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.head = Head(D + n_static)

    def forward(self, tokens, recency, mask, static):
        h = torch.relu(self.conv(self.steps(tokens, recency).transpose(1, 2))).transpose(1, 2)
        pooled = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return self.head(pooled, static)


class GRUEncoder(nn.Module):
    """Week 19: a clipboard that walks the events; read it after the most recent one."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.steps = Steps()
        self.rnn = nn.GRU(D, D, batch_first=True)
        self.head = Head(D + n_static)

    def forward(self, tokens, recency, mask, static):
        out, _ = self.rnn(self.steps(tokens, recency))
        return self.head(out[:, -1, :], static)  # left-padded: the last step is the latest event


class AttentionEncoder(nn.Module):
    """Week 20: every event looks at every other; padding is masked out, positions are learned."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.steps = Steps()
        self.pos = nn.Embedding(MAX_LEN, D)
        layer = nn.TransformerEncoderLayer(D, nhead=2, dim_feedforward=32, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = Head(D + n_static)

    def forward(self, tokens, recency, mask, static):
        x = self.steps(tokens, recency) + self.pos(torch.arange(tokens.shape[1]))
        h = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled, static)


ENCODERS = {
    "mlp, static only": StaticOnly,
    "cnn + static": ConvEncoder,
    "gru + static": GRUEncoder,
    "transformer + static": AttentionEncoder,
}


def run(seeds: tuple[int, ...] = (0, 1, 2), epochs: int = 20, n_train: int = 8000):
    train, test = bakeoff_data(n_train=n_train)
    return bakeoff(ENCODERS, train, test, seeds=seeds, epochs=epochs)


def main() -> None:
    table = run()
    print(table.to_string())
    best_net = table.drop(index="gbt (week 13)")["pr_auc"].idxmax()
    print(f"\nbest network: {best_net}; GBT PR-AUC {table.loc['gbt (week 13)', 'pr_auc']:.4f} "
          f"vs {table.loc[best_net, 'pr_auc']:.4f}")


if __name__ == "__main__":
    main()
