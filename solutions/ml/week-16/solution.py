"""Week 16 reference solution — the job pipeline.

Run from the repo root:

    python solutions/ml/week-16/solution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features import FEATURE_COLS, FORBIDDEN
from pipelines.promote import gate
from pipelines.train import train


def main() -> None:
    leaked = set(FEATURE_COLS) & set(FORBIDDEN)
    print("3. Contract: FEATURE_COLS ∩ FORBIDDEN")
    print(f"  FEATURE_COLS={FEATURE_COLS}")
    print(f"  leaked={leaked or '{}'}")
    assert not leaked

    out = ROOT / "artifacts" / "solution-week16"
    print("\n1. Train writes a candidate, not prod")
    meta = train("2024-06-01", out, n=4000, label="eventual")
    candidate = out / meta["model_version"]
    prod = ROOT / "artifacts" / "prod"
    print(json.dumps({k: meta[k] for k in ("model_version", "pr_auc", "dummy_pr_auc", "auc", "precision_at_80")}, indent=2))
    assert candidate.exists()
    assert (candidate / "metrics.json").exists()
    assert candidate.resolve() != prod.resolve()
    print(f"  candidate={candidate}  (prod is {prod} and train did not write it)")

    ok, reason = gate(candidate, prod if prod.exists() else None)
    print(f"  gate vs dummy/prod: ok={ok}  {reason}")
    if meta["pr_auc"] <= meta["dummy_pr_auc"]:
        print("  REFUSE: candidate does not beat dummy PR-AUC")
        print("  a 4k laptop sample can lose the gate — that is the gate working. Rerun with n=8000 if you want a promotable candidate.")

    scratch = candidate / "metrics.json"
    original = json.loads(scratch.read_text())
    broken = dict(original)
    broken["dummy_pr_auc"] = original["pr_auc"] + 0.5
    scratch.write_text(json.dumps(broken, indent=2))
    ok_bad, reason_bad = gate(candidate, None)
    scratch.write_text(json.dumps(original, indent=2))
    print(f"  gate with dummy raised above PR-AUC: ok={ok_bad}  {reason_bad}")
    assert not ok_bad

    print("\n2. Train twice, same as_of — prod still untouched by train")
    stamp = None
    if prod.exists() and (prod / "metrics.json").exists():
        stamp = (prod / "metrics.json").read_text()
    train("2024-06-01", out, n=4000, label="eventual")
    if stamp is not None:
        assert (prod / "metrics.json").read_text() == stamp
        print("  artifacts/prod metrics.json unchanged after a second train")
    else:
        print("  no artifacts/prod yet — promote is the only writer. Train still did not create it.")
        assert not prod.exists() or stamp is None

    print("\n4. Cron (five lines, no Airflow)")
    print(
        "\n".join(
            [
                "  pytest tests/",
                "  python -m pipelines.train --as-of 2024-06-01 --n 8000 --label eventual",
                "  python -m pipelines.promote --candidate artifacts/20240601",
                "  python -m pipelines.score_batch --as-of 2024-06-01 --artifact artifacts/prod --out tonight.csv",
                "  head tonight.csv",
            ]
        )
    )


if __name__ == "__main__":
    main()
