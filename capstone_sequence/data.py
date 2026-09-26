"""Event sequences as of a morning, on the job path's backtest split.

Each at-risk customer becomes their last MAX_LEN events up to `as_of`,
oldest first and left-padded, so the final position is always the most
recent event. Two channels per step: the event type (a token) and how long
before `as_of` it happened (recency). Nothing after `as_of` gets in.
A customer with no events gets a single NONE token, so every row has at
least one real position (an all-padding row breaks attention masks).

The static features are the same FEATURE_COLS row the GBT sees, so a
network can only beat it by finding something *in the order of events*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib.course_data import find_data_dir
from pipelines.features import FEATURE_COLS, make_preprocessor
from pipelines.split import BACKTEST_HORIZON_DAYS, snapshot_split
from pipelines.train import _keep_all_positives

EVENT_TYPES = ["cancel", "click", "downgrade", "feature_use", "login", "page_view",
               "payment", "signup", "support_message", "upgrade"]
VOCAB = {name: i + 1 for i, name in enumerate(EVENT_TYPES)}  # 0 is padding
PAD = 0
NONE = len(EVENT_TYPES) + 1  # "no events before as_of" is information, not an empty row
VOCAB_SIZE = NONE + 1
MAX_LEN = 16
RECENCY_SCALE = np.log1p(730.0)  # two years of history maps to ~1.0


@dataclass
class Split:
    frame: pd.DataFrame        # the as-of rows (for the GBT and for briefs)
    y: np.ndarray              # 0/1 labels over the horizon
    tokens: np.ndarray         # (N, MAX_LEN) int64, PAD on the left
    recency: np.ndarray        # (N, MAX_LEN) float32, 0 where padded
    mask: np.ndarray           # (N, MAX_LEN) bool, True on real events
    static: np.ndarray         # (N, n_static) float32, FEATURE_COLS through make_preprocessor


def load_events(data: Path | None = None) -> pd.DataFrame:
    data = Path(data) if data is not None else find_data_dir()
    return pd.read_csv(data / "user_events.csv", usecols=["user_id", "event_type", "timestamp"],
                       parse_dates=["timestamp"])


def sequences(user_ids: pd.Series, as_of: pd.Timestamp, events: pd.DataFrame,
              max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokens, recency, and mask for `user_ids`, using only events at or before `as_of`."""
    past = events[(events["timestamp"] <= as_of) & events["user_id"].isin(set(user_ids))]
    past = past.sort_values(["user_id", "timestamp"])
    past = past.groupby("user_id", sort=False).tail(max_len)
    days = (as_of - past["timestamp"]).dt.total_seconds() / 86400
    past = past.assign(tok=past["event_type"].map(VOCAB).fillna(PAD).astype(np.int64),
                       rec=(np.log1p(days.clip(lower=0)) / RECENCY_SCALE).astype(np.float32))

    row_of = {uid: i for i, uid in enumerate(user_ids)}
    tokens = np.full((len(user_ids), max_len), PAD, dtype=np.int64)
    recency = np.zeros((len(user_ids), max_len), dtype=np.float32)
    for uid, group in past.groupby("user_id", sort=False):
        k = len(group)
        tokens[row_of[uid], max_len - k:] = group["tok"].to_numpy()
        recency[row_of[uid], max_len - k:] = group["rec"].to_numpy()
    silent = (tokens == PAD).all(axis=1)
    tokens[silent, -1] = NONE
    recency[silent, -1] = 1.0
    return tokens, recency, tokens != PAD


def bakeoff_data(as_of: str | pd.Timestamp = "2024-06-01", horizon_days: int = BACKTEST_HORIZON_DAYS,
                 n_train: int | None = 8000, events: pd.DataFrame | None = None) -> tuple[Split, Split]:
    """(train, test): train is the snapshot `horizon_days` earlier, negatives downsampled
    to `n_train` rows for a laptop; test is the full `as_of` snapshot, never downsampled."""
    as_of = pd.Timestamp(as_of)
    events = events if events is not None else load_events()
    train_df, y_train, test_df, y_test = snapshot_split(as_of, horizon_days=horizon_days)
    train_df, y_train = _keep_all_positives(train_df, y_train, n_train)

    prep = make_preprocessor().fit(train_df[FEATURE_COLS])
    splits = []
    for frame, y, when in ((train_df, y_train, as_of - pd.Timedelta(days=horizon_days)), (test_df, y_test, as_of)):
        frame = frame.reset_index(drop=True)
        tokens, recency, mask = sequences(frame["user_id"], when, events)
        static = np.asarray(prep.transform(frame[FEATURE_COLS]), dtype=np.float32)
        splits.append(Split(frame, np.asarray(y, dtype=np.float32), tokens, recency, mask, static))
    return splits[0], splits[1]
