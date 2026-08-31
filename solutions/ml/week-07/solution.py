"""Week 07 reference solution — a score, then a threshold.

Run from the repo root:

    python solutions/ml/week-07/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import build_features

NUMERIC = ["mrr", "tenure_so_far", "log_usage", "features_adopted", "total_events", "n_support"]
CATEGORICAL = ["plan_type"]


def make_pipe(numeric: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    forest = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2)
    return Pipeline([("prep", prep), ("model", forest)])


def main() -> None:
    df = build_features(as_of="2024-06-01")
    X = df[NUMERIC + CATEGORICAL]
    y = df["is_churned"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = make_pipe(NUMERIC)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_np = y_test.to_numpy()

    print("1. Capacity budget — precision at 100 calls")
    order = np.argsort(-proba)
    top = order[:100]
    hits = int(y_np[top].sum())
    print(f"  flagged=100  actually churned={hits}  precision@100={hits / 100:.3f}")

    print("\n2. Threshold sweep")
    print(f"  {'cut':>6} {'flagged':>8} {'prec':>8} {'rec':>8}")
    ship = None
    for cut in np.arange(0.1, 1.0, 0.1):
        pred = proba >= cut
        flagged = int(pred.sum())
        prec = precision_score(y_np, pred, zero_division=0)
        rec = recall_score(y_np, pred, zero_division=0)
        print(f"  {cut:6.1f} {flagged:8d} {prec:8.3f} {rec:8.3f}")
        if ship is None and 50 <= flagged <= 150:
            ship = cut
    k = min(100, len(proba))
    cut_budget = float(np.partition(proba, -k)[-k])
    print(
        f"  circled for a 100-call desk: score cut≈{cut_budget:.4f} "
        f"(not 0.5 — that flagged {int((proba >= 0.5).sum())}; "
        f"sweep row to staff ≈ {ship})"
    )

    print("\n3. Ablation — drop tenure_so_far")
    auc_full = roc_auc_score(y_np, proba)
    numeric_ablate = [c for c in NUMERIC if c != "tenure_so_far"]
    ablate = make_pipe(numeric_ablate)
    ablate.fit(X_train[numeric_ablate + CATEGORICAL], y_train)
    auc_ab = roc_auc_score(y_test, ablate.predict_proba(X_test[numeric_ablate + CATEGORICAL])[:, 1])
    print(f"  AUC full={auc_full:.3f}  without tenure_so_far={auc_ab:.3f}  drop={auc_full - auc_ab:.3f}")
    print("  tenure_so_far is powerful and a bit circular for brand-new users; lifetime tenure_days is the real leak (Week 8).")


if __name__ == "__main__":
    main()
