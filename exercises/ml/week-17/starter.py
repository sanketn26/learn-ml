"""Week 17 exercises — You Are On-Call (and a Ticket Bot).

Run from the repo root:

    python exercises/ml/week-17/starter.py

Fill in each task_N below (delete the `raise NotImplementedError`). Every run
checks what your tasks return and prints ✓ / ✗ / · (not started) per task.
Task 4 is checked by `pytest tests/test_eval_router.py` once you edit the golden file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.checks import Task, expect, run  # noqa: E402
from pipelines.features import build_features  # noqa: E402

PROD = ROOT / "artifacts" / "prod"


def task_2():
    """Return get_churn_score(user_id) -> dict: predict()'s response, or {"error": ...} — never an exception."""
    raise NotImplementedError


def task_3() -> int:
    """What eval.router.evaluate() returns: the number of golden tickets that failed."""
    raise NotImplementedError


# ---- checks: read them if you are stuck; they say what "right" means ------------

def check_2(get_churn_score) -> None:
    unknown = get_churn_score("user_does_not_exist")
    expect(isinstance(unknown, dict) and "error" in unknown, "an unknown id should return {'error': ...}, not raise or guess")
    known_id = build_features(n=None)["user_id"].iloc[0]
    answer = get_churn_score(known_id)
    if (PROD / "model.joblib").exists():
        expect({"churn_score", "flag_for_cs", "model_version"} <= set(answer), f"with a prod model, return predict()'s dict; got {answer}")
    else:
        expect("error" in answer, "no artifacts/prod yet: return an error dict that says so (then run Week 16's train + promote)")


def check_3(failures) -> None:
    expect(failures == 0, f"{failures} golden ticket(s) failed — run `python -m eval.router` to see which")


CHECKS = [
    Task("1. Incident write-up", None, None, writeup=True),
    Task("2. Score as a tool", task_2, check_2),
    Task("3. Golden file", task_3, check_3),
    Task("4. Injection (pytest tests/test_eval_router.py)", None, None, writeup=True),
]

if __name__ == "__main__":
    run(CHECKS)
