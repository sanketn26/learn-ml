"""Deep-learning capstone — does the order of events beat the row?

Run from the repo root:

    python exercises/ml/capstone-sequence/starter.py

Write the three encoders. The runner bake-offs every finished encoder against
the Week-13 GBT and a static-only control, over three seeds. CPU, ~1 minute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from capstone_sequence.data import MAX_LEN, PAD, VOCAB_SIZE, bakeoff_data  # noqa: F401
from capstone_sequence.harness import bakeoff

D = 16


# --- provided: the parts every encoder shares --------------------------------

class Steps(nn.Module):
    """Event token + recency → one D-dim vector per step."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.mix = nn.Linear(D + 1, D)

    def forward(self, tokens, recency):
        return torch.relu(self.mix(torch.cat([self.embed(tokens), recency.unsqueeze(-1)], dim=-1)))


class Head(nn.Module):
    """Concatenate whatever you pass → one logit."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, *parts):
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class StaticOnly(nn.Module):
    """The control: the same head, no sequence."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.head = Head(n_static)

    def forward(self, tokens, recency, mask, static):
        return self.head(static)


# --- yours: every forward is (tokens, recency, mask, static) -> logits (B,) ----

class ConvEncoder(nn.Module):
    """Week 18. TODO 1: Conv1d over Steps, max-pool over *real* positions only, Head(D + n_static)."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        raise NotImplementedError("encoder 1: ConvEncoder")


class GRUEncoder(nn.Module):
    """Week 19. TODO 2: GRU over Steps; which position holds the latest event?"""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        raise NotImplementedError("encoder 2: GRUEncoder")


class AttentionEncoder(nn.Module):
    """Week 20. TODO 3: positions + a 1-layer TransformerEncoder with a padding mask; masked mean-pool."""

    def __init__(self, n_static: int) -> None:
        super().__init__()
        raise NotImplementedError("encoder 3: AttentionEncoder")


def main() -> None:
    train, test = bakeoff_data()
    encoders = {"mlp, static only": StaticOnly}
    for name, cls in (("cnn + static", ConvEncoder), ("gru + static", GRUEncoder),
                      ("transformer + static", AttentionEncoder)):
        try:
            cls(train.static.shape[1])
        except NotImplementedError as todo:
            print(f"skipping {name}: {todo}")
            continue
        encoders[name] = cls
    print(bakeoff(encoders, train, test, seeds=(0, 1, 2)).to_string())
    # TODO 5: your verdict, in two sentences, next to these numbers


if __name__ == "__main__":
    main()
