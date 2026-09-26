"""Gate for the deep-learning capstone (docs/ml/capstone-sequence.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone_sequence.data import NONE, PAD, bakeoff_data, load_events, sequences
from pipelines.split import snapshot_split

ROOT = Path(__file__).resolve().parent.parent
AS_OF = pd.Timestamp("2024-06-01")


@pytest.fixture(scope="module")
def splits():
    return bakeoff_data(n_train=2000)


def test_sequences_never_see_the_future():
    events = load_events()
    uid = events.loc[events["timestamp"] > AS_OF, "user_id"].iloc[0]
    tokens, recency, mask = sequences(pd.Series([uid]), AS_OF, events)
    n_past = int((events["user_id"].eq(uid) & (events["timestamp"] <= AS_OF)).sum())
    assert mask.sum() == max(min(n_past, tokens.shape[1]), 1)
    assert (recency[mask] >= 0).all()


def test_left_padding_keeps_the_latest_event_last(splits):
    _, test = splits
    assert test.mask[:, -1].all()                     # every row ends on a real position
    first_real = test.mask.argmax(axis=1)
    assert all(test.mask[i, first_real[i]:].all() for i in range(0, len(test.y), 997))
    assert (test.tokens[~test.mask] == PAD).all()


def test_silent_customers_get_a_none_token(splits):
    _, test = splits
    silent = test.mask.sum(axis=1) == 1
    assert (test.tokens[silent, -1] != PAD).all()
    assert (test.tokens == NONE).any()


def test_test_split_is_the_real_population(splits):
    train, test = splits
    _, y_train, test_df, y_test = snapshot_split(AS_OF)
    assert len(test.y) == len(test_df) and test.y.mean() == pytest.approx(y_test.mean())  # never downsampled
    assert train.y.sum() == y_train.sum()                                                 # every training positive kept
    assert train.static.shape[1] == test.static.shape[1]


def _solution():
    path = ROOT / "solutions" / "ml" / "capstone-sequence" / "solution.py"
    spec = importlib.util.spec_from_file_location("capstone_sequence_solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_bakeoff_runs_every_encoder_against_the_gbt(splits):
    from capstone_sequence.harness import bakeoff

    solution = _solution()
    train, test = splits
    table = bakeoff(solution.ENCODERS, train, test, seeds=(0,), epochs=2)
    assert list(table.index) == ["gbt (week 13)", *solution.ENCODERS]
    assert np.isfinite(table[["auc", "pr_auc"]].to_numpy()).all()
    assert table.loc["gbt (week 13)", "pr_lift"] > 1          # the bar is above chance


def test_the_bag_control_cannot_see_order(splits):
    import torch

    solution = _solution()
    _, test = splits
    rows = slice(0, 256)
    tokens, recency = test.tokens[rows].copy(), test.recency[rows].copy()
    mask, static = test.mask[rows], test.static[rows]
    rng = np.random.default_rng(0)
    shuffled_t, shuffled_r = tokens.copy(), recency.copy()
    for i in range(len(tokens)):
        real = np.flatnonzero(mask[i])
        order = rng.permutation(real)
        shuffled_t[i, real], shuffled_r[i, real] = tokens[i, order], recency[i, order]
    torch.manual_seed(0)
    bag = solution.BagOfEvents(static.shape[1]).eval()
    as_tensors = lambda t, r: (torch.from_numpy(t), torch.from_numpy(r), torch.from_numpy(mask), torch.from_numpy(static))
    with torch.no_grad():
        assert torch.allclose(bag(*as_tensors(tokens, recency)), bag(*as_tensors(shuffled_t, shuffled_r)), atol=1e-6)
