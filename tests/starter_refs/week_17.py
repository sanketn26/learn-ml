from pathlib import Path

from eval.router import evaluate
from pipelines.contract import load_artifact, predict
from pipelines.features import FEATURE_COLS, build_features

PROD = Path(__file__).resolve().parents[2] / "artifacts" / "prod"


def task_2():
    def get_churn_score(user_id):
        if not (PROD / "model.joblib").exists():
            return {"error": "no artifacts/prod — run Week 16 train + promote first"}
        frame = build_features(n=None)
        row = frame.loc[frame["user_id"] == user_id]
        if row.empty:
            return {"error": f"{user_id} is not an at-risk customer as of today"}
        payload = {k: (str(v) if k == "plan_type" else float(v)) for k, v in row[FEATURE_COLS].iloc[0].items()}
        return predict(payload, load_artifact(PROD))
    return get_churn_score


def task_3():
    return evaluate()
