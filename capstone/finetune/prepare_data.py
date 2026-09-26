"""Format teacher examples for LoRA SFT.

Reads capstone/data/*.jsonl, or regenerates them with the teacher if they
are missing. Every target command is re-validated against its spec *and*
the state it was chosen in — a row that fails its own contract never
reaches the trainer.

    python capstone/finetune/prepare_data.py --dry-run     # one night: fast pipeline check
    python capstone/finetune/prepare_data.py               # the full bank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capstone.cases import DEFAULT_NIGHTS
from capstone.prompt import messages_for, parse_call
from capstone.reliability import validate_call
from capstone.teacher import write_splits

DATA_DIR = ROOT / "capstone" / "data"
OUT_DIR = ROOT / "capstone" / "finetune" / "artifacts"
QUICK_NIGHTS = DEFAULT_NIGHTS[:1]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ensure_raw_splits(data_dir: Path, nights=DEFAULT_NIGHTS) -> dict[str, int]:
    needed = [data_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    if all(p.exists() and p.stat().st_size > 0 for p in needed):
        return {p.stem: len(_load_jsonl(p)) for p in needed}
    return write_splits(data_dir, nights=nights)


def format_row(row: dict) -> dict:
    """Chat-style example: system (instructions + command catalog), user (the state), assistant (one command)."""
    validate_call(row["call"], row["state"])
    return {
        "id": row["id"],
        "messages": messages_for(row["state"]) + [{"role": "assistant", "content": json.dumps(row["call"])}],
        "state": row["state"],
    }


def validate_formatted(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no examples to format")
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{row.get('id')}: expected system/user/assistant, got {roles}")
        call = parse_call(row["messages"][2]["content"])
        validate_call(call, row["state"])


def write_formatted(out_dir: Path, split: str, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.formatted.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Format capstone examples for LoRA.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="one night into a scratch data dir; checks the pipeline, not the bank")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if args.dry_run:
        data_dir, out_dir = out_dir / "dry-run-data", out_dir / "dry-run"
        for stale in data_dir.glob("*.jsonl"):
            stale.unlink()
    raw_counts = _ensure_raw_splits(data_dir, QUICK_NIGHTS if args.dry_run else DEFAULT_NIGHTS)
    print("raw examples:", raw_counts)

    for split in ("train", "val", "test"):
        formatted = [format_row(row) for row in _load_jsonl(data_dir / f"{split}.jsonl")]
        validate_formatted(formatted)
        path = write_formatted(out_dir, split, formatted)
        recovery = sum(r["id"].endswith("r") for r in formatted)
        print(f"  {split}: {len(formatted)} examples ({recovery} recover from a rejection) → {path}")
    if args.dry_run:
        print("dry-run: every example passed its spec, no model download")


if __name__ == "__main__":
    main()
