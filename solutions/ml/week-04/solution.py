"""Week 04 reference solution — charts that change a decision.

Run from the repo root:

    python solutions/ml/week-04/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
OUT = Path(__file__).resolve().parent


def main() -> None:
    df = load_customer_360()
    events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"])

    print("1. Adoption curve — mean features_adopted by plan")
    adopted = df.groupby("plan_type")["features_adopted"].mean().sort_values()
    print(adopted.round(2).to_string())
    winner = adopted.idxmax()
    fig, ax = plt.subplots(figsize=(7, 4))
    adopted.plot(kind="bar", ax=ax, color="#4f46e5")
    ax.set_ylabel("mean features adopted")
    ax.set_title(f"{winner} adopts the most features")
    ax.annotate(f"{adopted[winner]:.1f}", xy=(list(adopted.index).index(winner), adopted[winner]), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "adoption_by_plan.png", dpi=120)
    plt.close(fig)

    print("\n2. Region bars — churn rate, one region per user")
    def most_common(series: pd.Series):
        counts = series.dropna().value_counts()
        return counts.index[0] if len(counts) else pd.NA

    region = events.groupby("user_id")["region"].agg(most_common)
    mixed = df.merge(region.rename("region"), on="user_id", how="left")
    churn = mixed.groupby(mixed["region"].fillna("unknown"))["is_churned"].mean().sort_values()
    print(churn.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    churn.plot(kind="barh", ax=ax, color="#0f766e")
    ax.set_xlabel("churn rate")
    ax.set_title("Churn rate by most-common region")
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_region.png", dpi=120)
    plt.close(fig)

    print("\n3. Honest title — y-axis starts at 0")
    plan_churn = df.groupby("plan_type")["is_churned"].mean().sort_values()
    free = float(plan_churn.get("free", 0))
    paid = float(df.loc[df["plan_type"] != "free", "is_churned"].mean())
    ratio = (free / paid) if paid else float("nan")
    print(plan_churn.round(3).to_string())
    print(f"  free={free:.3f}  paid={paid:.3f}  ratio={ratio:.2f}×")
    fig, ax = plt.subplots(figsize=(7, 4))
    plan_churn.plot(kind="bar", ax=ax, color="#b45309")
    ax.set_ylim(0, max(0.25, float(plan_churn.max()) * 1.15))
    ax.set_ylabel("churn rate")
    ax.set_title(f"Free churn is ~{ratio:.1f}× paid")
    fig.tight_layout()
    fig.savefig(OUT / "honest_plan_churn.png", dpi=120)
    plt.close(fig)
    print(f"  wrote PNGs next to {OUT}")


if __name__ == "__main__":
    main()
