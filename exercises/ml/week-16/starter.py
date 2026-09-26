"""Week 16 exercises — The Job Pipeline.

Run from the repo root:

    python exercises/ml/week-16/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
Your tasks write under ARTIFACTS, exactly like the real job.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def task_1() -> tuple[tuple[bool, str], tuple[bool, str]]:
    """(gate() on the real candidate, gate() on a scratch copy whose dummy_pr_auc you rigged above its pr_auc)."""
    raise NotImplementedError


def task_2() -> tuple[str, str, str]:
    """Fingerprints of ARTIFACTS/"prod"/metrics.json: (before, after two train() calls, after promote())."""
    raise NotImplementedError


def task_3() -> set[str]:
    """FEATURE_COLS ∩ FORBIDDEN — the set your test asserts is empty."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_1(result) -> None:
    (ok_real, reason_real), (ok_rigged, _) = result
    expect(ok_real, f"the real candidate should pass the gate; it said: {reason_real}")
    expect(not ok_rigged, "the rigged copy must be refused — a dummy that beats the model never reaches prod")


def check_2(result) -> None:
    before, after_train, _after_promote = result
    expect(before == after_train, "train() changed prod. Train writes a candidate directory; only promote() may touch prod")


def check_3(overlap) -> None:
    expect(overlap == set(), f"forbidden columns in the model's input: {sorted(overlap)}")


CHECKS = [
    Task("1. Gate", task_1, check_1),
    Task("2. Train does not write prod", task_2, check_2),
    Task("3. One function", task_3, check_3),
    Task("4. Cron", None, None, writeup=True),
]

if __name__ == "__main__":
    run(CHECKS)
