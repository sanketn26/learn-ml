"""Train as of a day. Writes artifacts/<version>/ — does not touch prod."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.features import AS_OF_DEFAULT, CATEGORICAL, FEATURE_COLS, NUMERIC
from pipelines.split import BACKTEST_HORIZON_DAYS, snapshot_split

BUDGET = 80


def _threshold_for_budget(y: np.ndarray, scores: np.ndarray, budget: int) -> float:
    if len(scores) == 0:
        return 1.0
    k = min(budget, len(scores))
    return float(np.partition(scores, -k)[-k])


def _keep_all_positives(frame: pd.DataFrame, y: pd.Series, n: int, rng: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample negatives only. Dropping rares is how 0.1% events become 0 events."""
    if n is None or len(frame) <= n:
        return frame, y
    pos = frame.loc[y == 1]
    neg = frame.loc[y == 0]
    n_neg = max(n - len(pos), 0)
    if len(neg) > n_neg:
        neg = neg.sample(n_neg, random_state=rng)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=rng)
    return out, y.loc[out.index]


def train(as_of: str, out_dir: Path, n: int | None = 8000, horizon_days: int = BACKTEST_HORIZON_DAYS) -> dict:
    as_of_ts = pd.Timestamp(as_of)
    # Backtest, not a signup_date cut — a signup cut is a tenure cut (see pipelines/split.py).
    train_df, y_train, test_df, y_test = snapshot_split(as_of_ts, horizon_days=horizon_days, n=None)
    # Downsample negatives to fit on a laptop. Never the test set: every metric
    # below is quoted as "what the desk will see," so it runs on the real mix.
    train_df, y_train = _keep_all_positives(train_df, y_train, n)
    if y_train.nunique() < 2:
        raise RuntimeError(
            f"train set has one class (rate={float(y_train.mean())}). "
            "Use a larger --n, a longer --horizon-days, or a different --as-of."
        )

    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=40, learning_rate=0.1, max_depth=2, random_state=42
                ),
            ),
        ]
    )
    pipe.fit(train_df[FEATURE_COLS], y_train)
    scores = pipe.predict_proba(test_df[FEATURE_COLS])[:, 1]
    dummy = float(y_train.mean())
    dummy_ap = average_precision_score(y_test, np.full(len(y_test), dummy))
    ap = average_precision_score(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    threshold = _threshold_for_budget(y_test.to_numpy(), scores, BUDGET)
    flagged = scores >= threshold
    precision_at_budget = float(y_test.to_numpy()[np.argsort(-scores)[:BUDGET]].mean()) if len(y_test) else 0.0

    version = as_of_ts.strftime("%Y%m%d")
    meta = {
        "model_version": version,
        "as_of": str(as_of_ts.date()),
        "train_as_of": str((as_of_ts - pd.Timedelta(days=horizon_days)).date()),
        "label": f"churn within {horizon_days} days",
        "horizon_days": horizon_days,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "base_rate": float(y_test.mean()),
        "auc": round(float(auc), 4),
        "pr_auc": round(float(ap), 4),
        "dummy_pr_auc": round(float(dummy_ap), 4),
        "threshold": round(threshold, 4),
        "precision_at_80": round(precision_at_budget, 4),
        "flag_rate": round(float(flagged.mean()), 4),
        "features": FEATURE_COLS,
    }

    dest = Path(out_dir) / version
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": FEATURE_COLS}, dest / "model.joblib")
    (dest / "metrics.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CloudWave churn as of a day.")
    parser.add_argument("--as-of", default=str(AS_OF_DEFAULT.date()))
    parser.add_argument("--out", default=str(ROOT / "artifacts"))
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--horizon-days", type=int, default=BACKTEST_HORIZON_DAYS)
    args = parser.parse_args()
    meta = train(args.as_of, Path(args.out), n=args.n, horizon_days=args.horizon_days)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
