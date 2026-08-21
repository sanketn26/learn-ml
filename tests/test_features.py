"""Train path and serve path must build the same row."""

from __future__ import annotations

import pandas as pd

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, FORBIDDEN, build_features


def test_no_future_usage_after_as_of():
    as_of = AS_OF_DEFAULT
    df = build_features(as_of=as_of, n=500, at_risk_only=True)
    assert df["as_of"].nunique() == 1
    assert (df["signup_date"] <= as_of).all()
    assert not bool(df["already_churned"].any())


def test_lifetime_tenure_is_not_a_feature():
    df = build_features(n=200)
    assert "tenure_so_far" in df.columns
    assert "tenure_days" not in FEATURE_COLS
    leaked = set(FEATURE_COLS) & set(FORBIDDEN)
    assert not leaked


def test_matrix_columns_are_the_contract():
    df = build_features(n=50)
    X = df[FEATURE_COLS]
    assert list(X.columns) == FEATURE_COLS
    assert X.isna().sum().sum() == 0


def test_one_row_per_at_risk_user():
    as_of = AS_OF_DEFAULT
    df = build_features(as_of=as_of, n=None, at_risk_only=True)
    assert df["user_id"].is_unique
    assert (df["signup_date"] <= as_of).all()
    assert not bool(df["already_churned"].any())
