"""Week 08 reference solution — labels lie.

Run from the repo root:

    python solutions/ml/week-08/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.contract import validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon, label_eventual_churn

BUDGET = 80


def main() -> None:
    as_of = AS_OF_DEFAULT
    df = build_features(as_of=as_of, n=None, at_risk_only=True)

    print("0. Predict-first (check this against the sweep below)")
    print("  Raising the threshold 0.5 → 0.8 should raise precision and cut recall.")
    print("  Rare positives make that swing sharper: you run out of true hits fast.")

    print("\n1. Two rates on the same at-risk people")
    y_h = label_churn_in_horizon(df, as_of)
    labelled_h, y_h2 = drop_unlabelled(df, y_h)
    print(f"  horizon-30 rate={float(y_h2.mean()):.4f}  n={len(y_h2):,}  positives={int(y_h2.sum())}")
    print(f"  lifetime is_churned on those rows={float(labelled_h['is_churned'].mean()):.4f}")
    print("  legal at score time: the horizon label (or eventual-after-as_of). Not lifetime is_churned.")

    print("\n2. Censoring — observation_end = as_of + 10 days, horizon=30")
    short_end = as_of + pd_timedelta(days=10)
    y_c = label_churn_in_horizon(df, as_of, horizon_days=30, observation_end=short_end)
    print(f"  NaN (censored or already gone): {int(y_c.isna().sum()):,}")
    print(f"  observed cancels that survived as 1: {int((y_c == 1).sum()):,}")

    print("\n3. PR vs ROC on eventual labels")
    y_e = label_eventual_churn(df, as_of)
    frame, y = drop_unlabelled(df, y_e)
    X = frame[FEATURE_COLS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    dummy = float(y_train.mean())
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    dummy_pr = average_precision_score(y_test, np.full(len(y_test), dummy))
    order = np.argsort(-proba)[:BUDGET]
    prec80 = float(y_test.to_numpy()[order].mean())
    print(f"  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  dummy PR-AUC={dummy_pr:.3f}  precision@80={prec80:.3f}")
    print("  Monday email: PR-AUC vs dummy, plus precision@80. ROC-AUC is the ranking story, not the desk.")

    pred_05 = (proba >= 0.5).astype(int)
    pred_08 = (proba >= 0.8).astype(int)
    print(
        f"  thresh 0.5 prec={precision_score(y_test, pred_05, zero_division=0):.3f} "
        f"rec={recall_score(y_test, pred_05, zero_division=0):.3f}"
    )
    print(
        f"  thresh 0.8 prec={precision_score(y_test, pred_08, zero_division=0):.3f} "
        f"rec={recall_score(y_test, pred_08, zero_division=0):.3f}"
    )

    print("\n4. Forbidden payload key")
    demo = {k: (str(X_test.iloc[0][k]) if k == "plan_type" else float(X_test.iloc[0][k])) for k in FEATURE_COLS}
    try:
        validate({**demo, "churn_date": "2024-07-01"})
        print("  ERROR: validate should have raised")
    except Exception as exc:
        print(f"  validate extra key → {type(exc).__name__}: {exc}")

    print("\n5. Calibration glance")
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    print("  mean predicted vs fraction positive:")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"    pred={mp:.3f}  observed={fp:.3f}")
    print("  finance gets the score as a probability only if those sit on the diagonal. They usually do not.")


def pd_timedelta(days: int):
    import pandas as pd

    return pd.Timedelta(days=days)


if __name__ == "__main__":
    main()
