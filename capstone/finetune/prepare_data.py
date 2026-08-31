"""Format teacher trajectories for LoRA SFT.

Reads capstone/data/*.jsonl, or regenerates them with the synthetic teacher
if they are missing. Every tool_call is checked against capstone.tools and
reliability.validate_call.

    python capstone/finetune/prepare_data.py --dry-run
    python capstone/finetune/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capstone.reliability import RejectedCall, validate_call
from capstone.teacher import write_splits
from capstone.tools import TOOLS

DATA_DIR = ROOT / "capstone" / "data"
OUT_DIR = ROOT / "capstone" / "finetune" / "artifacts"

SYSTEM = (
    "You are CloudWave's coding specialist. Call exactly one of the five tools "
    "when the input matches, otherwise refuse with NO_TOOL. Never invent a tool name."
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ensure_raw_splits(data_dir: Path) -> dict[str, int]:
    needed = [data_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    if all(p.exists() and p.stat().st_size > 0 for p in needed):
        return {p.stem: len(_load_jsonl(p)) for p in needed}
    data_dir.mkdir(parents=True, exist_ok=True)
    return write_splits(data_dir)


def _output_text(tool_call: dict | None) -> str:
    if tool_call is None:
        return "NO_TOOL"
    return json.dumps({"name": tool_call["name"], "arguments": tool_call["arguments"]})


def format_row(row: dict) -> dict:
    """Chat-style example. Assistant content is a tool call or NO_TOOL."""
    call = row.get("tool_call")
    if call is not None:
        validate_call(call)
        if call["name"] not in TOOLS:
            raise RejectedCall(f"unknown tool {call['name']!r}")
    user = row.get("input") or ""
    if row.get("context"):
        user = f"{row['context']}\n\n{user}"
    return {
        "id": row.get("id"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _output_text(call)},
        ],
        "tool_call": call,
    }


def validate_formatted(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no trajectories to format")
    for row in rows:
        msgs = row["messages"]
        if len(msgs) != 3 or [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            raise ValueError(f"{row.get('id')}: expected system/user/assistant")
        assistant = msgs[2]["content"]
        if assistant == "NO_TOOL":
            continue
        call = json.loads(assistant)
        validate_call(call)


def write_formatted(out_dir: Path, split: str, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.formatted.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Format capstone trajectories for LoRA.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="validate only; still writes formatted files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    raw_counts = _ensure_raw_splits(data_dir)
    print("raw splits:", raw_counts)

    written = {}
    for split in ("train", "val", "test"):
        raw = _load_jsonl(data_dir / f"{split}.jsonl")
        formatted = [format_row(row) for row in raw]
        validate_formatted(formatted) if formatted else None
        path = write_formatted(out_dir, split, formatted)
        written[split] = {"n": len(formatted), "path": str(path)}
        print(f"  {split}: {len(formatted)} examples → {path}")

    n_tools = sum(
        1
        for split in ("train", "val", "test")
        for row in _load_jsonl(data_dir / f"{split}.jsonl")
        if row.get("tool_call")
    )
    print(f"tool-call rows (all splits): {n_tools}  surface={list(TOOLS)}")
    if args.dry_run:
        print("dry-run: schema ok, no model download")
    print("done", written)


if __name__ == "__main__":
    main()
