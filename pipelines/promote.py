"""Copy a candidate artifact to artifacts/prod only if it beats the gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gate(candidate: Path, prod: Path | None) -> tuple[bool, str]:
    cand = json.loads((candidate / "metrics.json").read_text())
    if cand["pr_auc"] < cand["dummy_pr_auc"]:
        return False, f"PR-AUC {cand['pr_auc']} < dummy {cand['dummy_pr_auc']}"
    if cand["auc"] < 0.52:
        return False, f"AUC {cand['auc']} is coin-flip"
    if prod is not None and (prod / "metrics.json").exists():
        prev = json.loads((prod / "metrics.json").read_text())
        if cand["pr_auc"] + 1e-6 < prev["pr_auc"]:
            return False, f"PR-AUC {cand['pr_auc']} < prod {prev['pr_auc']}"
    return True, "ok"


def promote(candidate: Path, prod: Path) -> None:
    ok, reason = gate(candidate, prod if prod.exists() else None)
    if not ok:
        raise SystemExit(f"REFUSED promote: {reason}")
    if prod.exists():
        shutil.rmtree(prod)
    shutil.copytree(candidate, prod)
    print(f"promoted {candidate.name} → {prod}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prod", default=str(ROOT / "artifacts" / "prod"))
    args = parser.parse_args()
    promote(Path(args.candidate), Path(args.prod))


if __name__ == "__main__":
    main()
