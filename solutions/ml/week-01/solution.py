"""Week 01 reference solution — NumPy on whole columns.

Run from the repo root:

    python solutions/ml/week-01/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir

DATA = find_data_dir()
SAMPLE_USERS = 400


def main() -> None:
    usage = pd.read_csv(DATA / "feature_usage.csv", usecols=["user_id", "feature_name", "usage_count"])

    print("1. Feature ranking — stats of *feature totals*, not row usage")
    totals = usage.groupby("feature_name", sort=False)["usage_count"].sum()
    arr = totals.to_numpy(dtype=float)
    print(f"  n_features={len(arr)}  mean={arr.mean():.1f}  median={np.median(arr):.1f}  p90={np.quantile(arr, 0.9):.1f}")

    print("\n2. Broadcasting on a sampled user × feature pivot")
    sample_ids = (
        usage["user_id"].drop_duplicates().sample(n=min(SAMPLE_USERS, usage["user_id"].nunique()), random_state=0)
    )
    pivot = (
        usage[usage["user_id"].isin(sample_ids)]
        .pivot_table(index="user_id", columns="feature_name", values="usage_count", aggfunc="sum", fill_value=0)
    )
    mat = pivot.to_numpy(dtype=float)
    row_means = mat.mean(axis=1, keepdims=True)
    normalized = mat / np.where(row_means == 0, 1.0, row_means)
    print(f"  mat {mat.shape}  /  row_means {row_means.shape}  →  {normalized.shape}")
    print(f"  row-mean of normalized (should be ~1): {normalized.mean(axis=1).mean():.3f}")

    print("\n3. Whale hunt — top 1% of users by total usage")
    per_user = usage.groupby("user_id", sort=False)["usage_count"].sum()
    cut = float(np.quantile(per_user.to_numpy(dtype=float), 0.99))
    whales = per_user[per_user >= cut]
    share = float(whales.sum() / per_user.sum())
    print(f"  cut={cut:.0f}  n_whales={len(whales)}  share_of_usage={share:.1%}")
    print("  sample ids:", list(whales.index[:8]))


if __name__ == "__main__":
    main()
