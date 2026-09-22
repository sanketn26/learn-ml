"""Week 15 reference solution — the pickle.

Run from the repo root:

    python solutions/ml/week-15/solution.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.contract import predict, validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon
from pipelines.split import snapshot_split

BUDGET = 80
HORIZON = 90
VERSION = "week15"


def _payload(row) -> dict:
    return {
        key: (str(row[key]) if key == "plan_type" else float(row[key]))
        for key in FEATURE_COLS
    }


def main() -> None:
    def fit_auc(x_tr, y_tr, x_te, y_te) -> tuple[Pipeline, float]:
        pipe = Pipeline(
            [
                ("prep", make_preprocessor()),
                ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
            ]
        )
        pipe.fit(x_tr[FEATURE_COLS], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(x_te[FEATURE_COLS])[:, 1])
        return pipe, float(auc)

    print("1. Three time walls, one GBT, 90-day horizon label")
    snap = build_features(as_of=AS_OF_DEFAULT, n=None)
    snap, y = drop_unlabelled(snap, label_churn_in_horizon(snap, AS_OF_DEFAULT, horizon_days=HORIZON))
    print(f"  snapshot rows={len(snap):,}  positives={int(y.sum())}  rate={float(y.mean()):.4f}")

    x_tr_s, x_te_s, y_tr_s, y_te_s = train_test_split(snap, y, test_size=0.2, random_state=42, stratify=y)
    _, auc_shuffle = fit_auc(x_tr_s, y_tr_s, x_te_s, y_te_s)

    cutoff = snap["signup_date"].quantile(0.80)
    early, late = snap[snap["signup_date"] <= cutoff], snap[snap["signup_date"] > cutoff]
    _, auc_signup = fit_auc(early, y.loc[early.index], late, y.loc[late.index])

    train_t, y_tr, test_t, y_te = snapshot_split(AS_OF_DEFAULT, horizon_days=HORIZON)
    time_pipe, auc_time = fit_auc(train_t, y_tr, test_t, y_te)

    print(f"  (a) shuffled    AUC={auc_shuffle:.3f}  — rows from the same day on both sides; a unit test")
    print(f"  (b) signup cut  AUC={auc_signup:.3f}  — tenure train {early['tenure_so_far'].min()}-{early['tenure_so_far'].max()}d, "
          f"test {late['tenure_so_far'].min()}-{late['tenure_so_far'].max()}d: a tenure cut, not a time wall")
    print(f"  (c) backtest    AUC={auc_time:.3f}  — train on {(AS_OF_DEFAULT - pd.Timedelta(days=HORIZON)).date()}, "
          f"score {AS_OF_DEFAULT.date()}: the production question")

    dest = ROOT / "artifacts" / VERSION
    dest.mkdir(parents=True, exist_ok=True)
    scores = time_pipe.predict_proba(test_t[FEATURE_COLS])[:, 1]
    k = min(BUDGET, len(scores))
    threshold = float(np.partition(scores, -k)[-k])
    meta = {
        "model_version": VERSION,
        "threshold": round(threshold, 4),
        "horizon_days": HORIZON,
        "auc_backtest": round(auc_time, 4),
        "auc_shuffled": round(auc_shuffle, 4),
        "auc_signup_cut": round(auc_signup, 4),
    }
    joblib.dump({"pipeline": time_pipe, "features": FEATURE_COLS}, dest / "model.joblib")
    (dest / "metrics.json").write_text(json.dumps(meta, indent=2))
    artifact = {"pipeline": time_pipe, "metrics": meta}

    print("\n2. predict() contract — 80 calls")
    demo = _payload(test_t.iloc[0])
    try:
        validate({**demo, "email": "ada@cloudwave.test"})
    except Exception as exc:
        print(f"  unknown key → {type(exc).__name__}: {exc}")
    t0 = time.perf_counter()
    lat = []
    out = None
    for i in range(80):
        row = test_t.iloc[i % len(test_t)]
        s = time.perf_counter()
        out = predict(_payload(row), artifact)
        lat.append(time.perf_counter() - s)
    elapsed = time.perf_counter() - t0
    arr = np.array(lat) * 1000
    print(f"  last response: {out}")
    print(f"  80 calls in {elapsed:.3f}s  p50={np.percentile(arr, 50):.2f}ms  p95={np.percentile(arr, 95):.2f}ms")

    print("\n3. Capacity, not 0.5")
    y_te_np = y_te.to_numpy()
    flagged = scores >= threshold
    prec_cap = float(y_te_np[np.argsort(-scores)[:k]].mean())
    pred_05 = scores >= 0.5
    prec_05 = float(y_te_np[pred_05].mean()) if pred_05.any() else 0.0
    rec_cap = float(y_te_np[np.argsort(-scores)[:k]].sum() / max(y_te_np.sum(), 1))
    rec_05 = float(((pred_05) & (y_te_np == 1)).sum() / max(y_te_np.sum(), 1))
    print(f"  budget cut={threshold:.3f}  flagged={int(flagged.sum())}  prec={prec_cap:.3f}  rec={rec_cap:.3f}")
    print(f"  0.5          flagged={int(pred_05.sum())}  prec={prec_05:.3f}  rec={rec_05:.3f}")

    print("\n4. Drift sketch — train snapshot vs today's snapshot")
    for col in ("mrr", "log_usage", "tenure_so_far"):
        a, b = float(train_t[col].mean()), float(test_t[col].mean())
        print(f"  {col:16s}  train_mean={a:.3f}  today_mean={b:.3f}  delta={b - a:+.3f}")
    print("  one sentence: tenure_so_far moves ~90 days by construction; mrr and usage moving is the world moving.")

    print("\n5. One-page write-up")
    print("  Time wall: a 90-day snapshot backtest. Shuffled AUC is the vanity number; a signup cut is a tenure cut.")
    print("  Beat a dummy (0.5 AUC / base-rate AP) or do not ship. Capacity=80, not threshold 0.5.")
    print("  Drift risk: new-plan or pricing changes shift mrr. Refused: 'this score is a probability' without calibration.")
    print(f"  artifact: {dest}")


if __name__ == "__main__":
    main()
