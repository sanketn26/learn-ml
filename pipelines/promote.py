"""Copy a candidate artifact to artifacts/prod only if it beats the gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.features import FEATURE_COLS

Holdout = tuple[pd.DataFrame, pd.Series]


def holdout_for(candidate: Path) -> Holdout:
    """The candidate's own matured test snapshot: the population both models are graded on."""
    from pipelines.split import snapshot_split

    meta = json.loads((candidate / "metrics.json").read_text())
    _, _, frame, y = snapshot_split(meta["as_of"], horizon_days=meta["horizon_days"])
    return frame, y


def _pr_auc(artifact: Path, holdout: Holdout) -> float:
    frame, y = holdout
    pipe = joblib.load(artifact / "model.joblib")["pipeline"]
    return float(average_precision_score(y, pipe.predict_proba(frame[FEATURE_COLS])[:, 1]))


def gate(candidate: Path, prod: Path | None, holdout: Holdout | None = None) -> tuple[bool, str]:
    """Refuse a candidate that loses to the dummy, or to prod *on the same holdout*.

    prod's metrics.json holds its PR-AUC on its own backtest — a different month, a
    different base rate, a different mix of easy and hard customers. Comparing the
    candidate's fresh number to that stored one promotes on prevalence, not quality.
    So both models are re-scored on the candidate's matured holdout.
    """
    cand = json.loads((candidate / "metrics.json").read_text())
    if cand["pr_auc"] < cand["dummy_pr_auc"]:
        return False, f"PR-AUC {cand['pr_auc']} < dummy {cand['dummy_pr_auc']}"
    if cand["auc"] < 0.52:
        return False, f"AUC {cand['auc']} is coin-flip"
    if prod is not None and (prod / "metrics.json").exists():
        prev = json.loads((prod / "metrics.json").read_text())
        if prev.get("horizon_days") != cand.get("horizon_days"):
            # PR-AUC moves with the base rate: a 90-day model's 0.13 and a 30-day
            # model's 0.05 are answers to different questions, not better and worse.
            return False, (f"prod answers a different question (horizon {prev.get('horizon_days')}d vs "
                           f"{cand.get('horizon_days')}d); review and promote by hand")
        if not (prod / "model.joblib").exists():
            return False, "prod has no model.joblib to re-score; review and promote by hand"
        holdout = holdout if holdout is not None else holdout_for(candidate)
        cand_ap, prod_ap = _pr_auc(candidate, holdout), _pr_auc(prod, holdout)
        where = f"same holdout, {len(holdout[1]):,} customers"
        if cand_ap + 1e-6 < prod_ap:
            return False, f"PR-AUC {cand_ap:.4f} < prod {prod_ap:.4f} ({where})"
        return True, f"ok: PR-AUC {cand_ap:.4f} vs prod {prod_ap:.4f} ({where})"
    return True, "ok"


def promote(candidate: Path, prod: Path) -> None:
    ok, reason = gate(candidate, prod if prod.exists() else None)
    if not ok:
        raise SystemExit(f"REFUSED promote: {reason}")
    if prod.exists():
        shutil.rmtree(prod)
    shutil.copytree(candidate, prod)
    print(f"promoted {candidate.name} → {prod} ({reason})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prod", default=str(ROOT / "artifacts" / "prod"))
    args = parser.parse_args()
    promote(Path(args.candidate), Path(args.prod))


if __name__ == "__main__":
    main()
