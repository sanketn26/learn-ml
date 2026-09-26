import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from pipelines.features import FEATURE_COLS, FORBIDDEN
from pipelines.promote import gate, promote
from pipelines.train import train

ARTIFACTS = Path(tempfile.mkdtemp(prefix="week16-ref-"))


def _fp(d):
    f = d / "metrics.json"
    return hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.exists() else "missing"


def task_1():
    meta = train("2024-06-01", ARTIFACTS)
    cand = ARTIFACTS / meta["model_version"]
    scratch = ARTIFACTS / "scratch-worse"
    shutil.copytree(cand, scratch, dirs_exist_ok=True)
    m = json.loads((scratch / "metrics.json").read_text())
    m["dummy_pr_auc"] = m["pr_auc"] + 0.05
    (scratch / "metrics.json").write_text(json.dumps(m))
    return gate(cand, None), gate(scratch, None)


def task_2():
    prod = ARTIFACTS / "prod"
    before = _fp(prod)
    for _ in range(2):
        meta = train("2024-06-01", ARTIFACTS)
    after = _fp(prod)
    promote(ARTIFACTS / meta["model_version"], prod)
    return before, after, _fp(prod)


def task_3():
    return set(FEATURE_COLS) & set(FORBIDDEN)
